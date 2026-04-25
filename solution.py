"""
AgenticML solution module.

This file is what you edit for base-model experiments. The harness in ``harness/``
runs cross-validation over the dataset and calls ``fit_predict`` once per fold,
passing only the training and validation slices of that fold. You never see
the other folds, the test set, or the CV indices themselves.

Contract
--------
Define exactly two things at module scope:

  HYPOTHESIS : str
      A one-line plain string literal describing what this attempt tries.
      Used as the git commit message and MLflow tag. Must be a literal — it
      is read via ast.parse without executing the module.

  fit_predict(X_train, y_train, X_val) -> np.ndarray
      Train your model on (X_train, y_train) and return predictions on X_val.
      Must return an np.ndarray of shape (len(X_val), n_classes) with
      probabilities (rows sum to 1).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tabm

HYPOTHESIS = "TabM default config (k=32, 3 blocks, d=512) with AdamW + early stopping"


def fit_predict(X_train, y_train, X_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    # One-hot encode categoricals, fill NaN
    X_tr = pd.get_dummies(X_train).fillna(0).astype(np.float32)
    X_vl = pd.get_dummies(X_val).fillna(0).astype(np.float32)
    X_vl = X_vl.reindex(columns=X_tr.columns, fill_value=0)

    # Standardize
    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr.values)
    X_vl_np = scaler.transform(X_vl.values)

    n_features = X_tr_np.shape[1]

    # Internal validation split for early stopping (10%)
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_tr_np, y_enc, test_size=0.1, random_state=42, stratify=y_enc
    )

    def to_tensor(X, y=None):
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        if y is None:
            return Xt
        yt = torch.tensor(y, dtype=torch.long, device=device)
        return Xt, yt

    X_fit_t, y_fit_t = to_tensor(X_fit, y_fit)
    X_es_t, y_es_t = to_tensor(X_es, y_es)
    X_vl_t = to_tensor(X_vl_np)

    # Class weights for imbalanced High class
    class_counts = np.bincount(y_enc)
    class_weights = torch.tensor(
        len(y_enc) / (n_classes * class_counts), dtype=torch.float32, device=device
    )

    # Build model
    model = tabm.TabM.make(
        n_num_features=n_features,
        d_out=n_classes,
        # defaults: k=32, n_blocks=3, d_block=512, dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    batch_size = 1024
    dataset = TensorDataset(X_fit_t, y_fit_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(200):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            # output: (batch_size, k, n_classes)
            logits = model(x_num=xb)
            # average ensemble predictions, then cross-entropy
            log_probs = F.log_softmax(logits, dim=-1).mean(dim=1)
            loss = F.nll_loss(log_probs, yb, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Early stopping check
        model.eval()
        with torch.no_grad():
            val_logits = model(x_num=X_es_t)
            val_log_probs = F.log_softmax(val_logits, dim=-1).mean(dim=1)
            val_loss = F.nll_loss(val_log_probs, y_es_t, weight=class_weights).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Predict
    model.eval()
    chunk = 4096
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_vl_t), chunk):
            xb = X_vl_t[i : i + chunk]
            logits = model(x_num=xb)  # (chunk, k, n_classes)
            probs = F.softmax(logits, dim=-1).mean(dim=1)  # (chunk, n_classes)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)

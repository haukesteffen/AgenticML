"""Thin wrappers around the `kaggle` Python package (v2.x).

Kaggle is imported lazily so `harness run` works without Kaggle creds
configured. Authentication happens on first API call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubmissionRef:
    description: str
    file_name: str
    date: str | None
    ref: str | None


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def submit(competition_slug: str, csv_path: Path, message: str) -> SubmissionRef:
    api = _api()
    api.competition_submit(str(csv_path), message, competition_slug, quiet=True)

    submissions = api.competition_submissions(competition_slug) or []
    match = _find_submission(submissions, message)
    if match is None:
        raise RuntimeError(
            f"Submitted to {competition_slug} but could not locate the new submission "
            f"by description={message!r}"
        )
    return SubmissionRef(
        description=match.description or message,
        file_name=match.file_name or "",
        date=str(match.date) if match.date is not None else None,
        ref=match.ref,
    )


def poll_public_score(
    competition_slug: str,
    ref: SubmissionRef,
    timeout: int = 180,
    interval: int = 15,
) -> float | None:
    api = _api()
    deadline = time.monotonic() + timeout
    while True:
        submissions = api.competition_submissions(competition_slug) or []
        match = _find_submission(submissions, ref.description, preferred_ref=ref.ref)
        if match is not None:
            score = match.public_score
            if score not in (None, ""):
                try:
                    return float(score)
                except (TypeError, ValueError):
                    pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


def _find_submission(submissions, description: str, preferred_ref: str | None = None):
    matches = [s for s in submissions if s is not None and (s.description or "") == description]
    if not matches:
        return None
    if preferred_ref:
        for s in matches:
            if s.ref == preferred_ref:
                return s
    matches.sort(key=lambda s: str(s.date or ""), reverse=True)
    return matches[0]

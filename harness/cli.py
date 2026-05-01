from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harness", description="AgenticML experiment harness")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Execute an experiment run")

    sub.add_parser("promote", help="Run nested 5x5 on solution.py and log to the promoted experiment")

    status_parser = sub.add_parser("status", help="Show recent experiment runs")
    status_parser.add_argument("--limit", type=int, default=10, help="Max runs to show")
    status_parser.add_argument(
        "--experiment", default=None,
        help="Override experiment name. Use 'promoted' to view all promoted lanes.",
    )
    status_parser.add_argument(
        "--lane", default=None,
        help="Filter promoted runs by lane (e.g. 'v1_raw__LGBMClassifier').",
    )

    submit_parser = sub.add_parser(
        "submit", help="Build submission.csv from a run's test_predictions.npy and upload to Kaggle"
    )
    submit_parser.add_argument(
        "--run-id", default=None,
        help="MLflow run id to submit. Defaults to best improved run on current branch.",
    )
    submit_parser.add_argument(
        "--message", default=None,
        help="Kaggle submission description. Defaults to 'branch | sha | cv=... | hypothesis'.",
    )
    submit_parser.add_argument(
        "--branch", default=None,
        help="Look up the best run for this branch instead of the current working-tree branch.",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        from harness.runner import run
        run(args.config)
    elif args.command == "promote":
        from harness.promote import main as promote_main
        promote_main(args.config)
    elif args.command == "status":
        from harness.status import status
        status(args.config, args.limit, experiment=args.experiment, lane=args.lane)
    elif args.command == "submit":
        from harness.submit import submit
        submit(args.config, run_id=args.run_id, message=args.message, branch=args.branch)

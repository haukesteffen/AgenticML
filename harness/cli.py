from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harness", description="AgenticML experiment harness")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Execute an experiment run")

    status_parser = sub.add_parser("status", help="Show recent experiment runs")
    status_parser.add_argument("--limit", type=int, default=10, help="Max runs to show")

    args = parser.parse_args(argv)

    if args.command == "run":
        from harness.runner import run
        run(args.config)
    elif args.command == "status":
        from harness.status import status
        status(args.config, args.limit)

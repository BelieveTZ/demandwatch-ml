import argparse
import json
from pathlib import Path

from demandwatch.data import download


def main():
    parser = argparse.ArgumentParser(description="DemandWatch reproducible ML workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download attributed UCI source data")
    fetch.add_argument("--directory", type=Path, default=Path("data/raw"))
    train = commands.add_parser("train", help="Select, calibrate and evaluate CPU models")
    train.add_argument("--source", type=Path, default=Path("data/raw/hour.csv"))
    train.add_argument("--output", type=Path, default=Path("artifacts/current"))
    train.add_argument("--report", type=Path, default=Path("demandwatch/web/data/report.json"))
    serve = commands.add_parser("serve", help="Serve saved dashboard and optional local inference")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--artifacts", type=Path, default=Path("artifacts/current"))
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(args.directory), indent=2))
    elif args.command == "train":
        from demandwatch.training import train as run_training

        report = run_training(args.source, args.output, args.report)
        print(
            json.dumps({key: report[key] for key in ["run_id", "metrics", "intervals"]}, indent=2)
        )
    else:
        import uvicorn

        from demandwatch.api import create_app

        uvicorn.run(create_app(args.artifacts), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

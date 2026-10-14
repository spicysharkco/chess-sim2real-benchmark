"""Bounded CPU-only shard runner; each renderer is a checked subprocess."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .benchmark_io import split_rows
    from .benchmark_v2 import load_plan, render_commands
except ImportError:
    from benchmark_io import split_rows
    from benchmark_v2 import load_plan, render_commands


def run_shards(run, repo, cohort, playwright, browser, workers):
    if workers not in (1, 2, 3, 4):
        raise ValueError("render workers must be between one and four")
    plan = load_plan(run, repo)
    commands = render_commands(run, repo, cohort, playwright, browser)
    shards = [s for s in plan["shards"] if s["cohort"] == cohort]
    (run / "renders").mkdir(exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)

    def render(item):
        shard, command = item
        folder = run / "renders" / shard["name"]
        if folder.exists():
            raise FileExistsError(
                f"output already exists; do not reuse a partial run: {folder}"
            )
        with (run / "logs" / f"{shard['name']}.log").open("x") as log:
            subprocess.run(
                command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        if len(split_rows(folder)) != shard["count"]:
            raise ValueError(f"shard count mismatch: {shard['name']}")
        return {"shard_complete": shard["name"], "images": shard["count"]}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(render, item) for item in zip(shards, commands)]
        try:
            for future in as_completed(futures):
                print(json.dumps(future.result()), flush=True)
        except BaseException:
            # Do not kill unrelated processes; allow active own shards to finish,
            # cancel only work which this executor has not yet started.
            for future in futures:
                future.cancel()
            raise
    load_plan(run, repo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--cohort", required=True, choices=("benchmark", "adaptation"))
    parser.add_argument("--playwright", required=True)
    parser.add_argument("--browser", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_shards(
        args.run, args.repo, args.cohort, args.playwright, args.browser, args.workers
    )


if __name__ == "__main__":
    main()

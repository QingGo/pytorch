import argparse
import json
import logging
import multiprocessing
import multiprocessing.pool
import os
import subprocess
import sys
import time
import traceback
import torch  # noqa: F401 — must be imported before pytest to prevent rootdir shadowing
import pytest
from pathlib import Path


def _get_repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _find_test_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in path.rglob("test_*.py"):
                if child.is_file():
                    files.append(child)
        else:
            logging.error("%s is not a file or directory", path)
            sys.exit(1)
    files.sort()
    return files


def _collect_one_file(
    file_path: str, repo_root: str, extra_pytest_args: list[str]
) -> dict:
    tests: list[str] = []
    markers: set[str] = set()
    error: str | None = None

    class CollectorPlugin:
        @staticmethod
        def pytest_collection_finish(session):
            for item in session.items:
                tests.append(item.nodeid)
                for marker in item.iter_markers():
                    markers.add(marker.name)

        @staticmethod
        def pytest_collectreport(report):
            nonlocal error
            if report.failed:
                error = str(report.longrepr).splitlines()[-1]

    try:
        exit_code = pytest.main(
            ["--collect-only", "-q", f"--rootdir={repo_root}"]
            + extra_pytest_args
            + [file_path],
            plugins=[CollectorPlugin()],
        )
        if not error and not tests and exit_code not in (0, 5):
            error = f"pytest exit {exit_code}"
    except SystemExit:
        pass
    except Exception:
        error = traceback.format_exc().splitlines()[-1]

    return {
        "file": os.path.relpath(file_path, repo_root),
        "tests": tests,
        "markers": sorted(markers),
        "error": error,
    }


def _collect_from_batch(args: tuple[list[str], str, list[str]]) -> list[dict]:
    file_paths, repo_root, extra_pytest_args = args

    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        return [
            _collect_one_file(fp, repo_root, extra_pytest_args) for fp in file_paths
        ]
    finally:
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def _default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def _make_batches(
    files: list[Path], jobs: int, min_batch_size: int = 20
) -> list[list[str]]:
    num_batches = min(jobs, max(1, len(files) // min_batch_size))
    batch_size = max(1, -(-len(files) // num_batches))
    return [
        [str(f) for f in files[i : i + batch_size]]
        for i in range(0, len(files), batch_size)
    ]


def _run_batched(
    files: list[Path],
    repo_root: Path,
    jobs: int,
    label: str,
    extra_pytest_args: list[str] | None = None,
) -> list[dict]:
    extra_pytest_args = extra_pytest_args or []
    batches = _make_batches(files, jobs)

    logging.info(
        "collecting %s from %d files in %d batches...",
        label,
        len(files),
        len(batches),
    )
    ctx = multiprocessing.get_context("spawn")
    batch_timeout = 120
    repo_root_str = str(repo_root)

    def _collect_with_timeout(
        pool: multiprocessing.pool.Pool,
        batches: list[list[str]],
        total_label: int,
    ) -> list[dict]:
        results: list[dict] = []
        batch_args = [(batch, repo_root_str, extra_pytest_args) for batch in batches]
        it = pool.imap_unordered(_collect_from_batch, batch_args)
        while len(results) < total_label:
            try:
                batch_results = it.next(timeout=batch_timeout)
                results.extend(batch_results)
                width = len(str(total_label))
                batch_failed = sum(
                    1 for r in batch_results if r["error"] and not r["tests"]
                )
                msg = f"{len(results):{width}d}/{total_label} files processed"
                if batch_failed:
                    msg += f" ({batch_failed} failed)"
                logging.info(msg)
            except StopIteration:
                break
            except multiprocessing.TimeoutError:
                pool.terminate()
                logging.warning(
                    "batch timed out after %ds, skipping %d remaining files",
                    batch_timeout,
                    total_label - len(results),
                )
                processed = {r["file"] for r in results}
                for batch in batches:
                    for fp in batch:
                        rel = os.path.relpath(fp, repo_root_str)
                        if rel not in processed:
                            results.append(
                                {
                                    "file": rel,
                                    "tests": [],
                                    "markers": [],
                                    "error": "collection timed out",
                                }
                            )
                break
        return results

    with ctx.Pool(processes=jobs) as pool:
        results = _collect_with_timeout(pool, batches, len(files))

    max_retries = 1
    for attempt in range(max_retries):
        failed = [r for r in results if r["error"] and not r["tests"]]
        if not failed:
            break
        logging.info(
            "%d/%d retry: re-batching %d failed files...",
            attempt + 1,
            max_retries,
            len(failed),
        )
        failed_set = {r["file"] for r in failed}
        failed_files = [repo_root / r["file"] for r in failed]
        retry_batches = [[str(f)] for f in failed_files]
        with ctx.Pool(processes=jobs) as retry_pool:
            retried_results = _collect_with_timeout(
                retry_pool, retry_batches, len(failed_files)
            )
        retried = {r["file"]: r for r in retried_results}
        results = [
            retried[r["file"]] if r["file"] in failed_set else r for r in results
        ]

    for result in results:
        if result["error"]:
            logging.warning("%s: %s", result["file"], result["error"])

    return results


def cmd_collect(args: argparse.Namespace) -> None:
    repo_root = _get_repo_root()
    paths = [p if p.is_absolute() else repo_root / p for p in args.paths]

    start = time.monotonic()

    files = _find_test_files(paths)
    if not files:
        logging.warning("no test files found")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"tests": {}}))
        return

    extra_pytest_args: list[str] = []
    if args.marker:
        extra_pytest_args.extend(["-m", args.marker])

    results = _run_batched(files, repo_root, args.jobs, "tests", extra_pytest_args)

    tests: dict[str, list[str]] = {}
    for result in results:
        if result["tests"]:
            tests[result["file"]] = result["tests"]

    elapsed = time.monotonic() - start
    test_count = sum(len(v) for v in tests.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"tests": dict(sorted(tests.items()))}))
    logging.info(
        "%d tests collected from %d files in %.2fs: %s",
        test_count,
        len(tests),
        elapsed,
        args.output,
    )


def cmd_shard(args: argparse.Namespace) -> None:
    data = json.loads(args.input.read_text())
    all_tests: list[str] = []
    for tests in data["tests"].values():
        all_tests.extend(tests)
    all_tests.sort()

    shard_tests = all_tests[args.shard_id :: args.num_shards]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(shard_tests) + ("\n" if shard_tests else ""))
    logging.info(
        "shard %d/%d: %d tests (of %d total)",
        args.shard_id + 1,
        args.num_shards,
        len(shard_tests),
        len(all_tests),
    )


def cmd_list_markers(args: argparse.Namespace) -> None:
    repo_root = _get_repo_root()
    paths = [p if p.is_absolute() else repo_root / p for p in args.paths]

    start = time.monotonic()

    files = _find_test_files(paths)
    if not files:
        logging.warning("no test files found")
        print(json.dumps({}))
        return

    results = _run_batched(files, repo_root, args.jobs, "markers")

    markers: dict[str, list[str]] = {}
    for result in results:
        if result["markers"]:
            markers[result["file"]] = result["markers"]

    elapsed = time.monotonic() - start
    logging.info(
        "markers collected from %d/%d files in %.2fs",
        len(markers),
        len(files),
        elapsed,
    )
    for file, file_markers in sorted(markers.items()):
        print(f"{file}: {', '.join(file_markers)}")


def main() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("\033[2mzippo %(asctime)s\033[0m %(message)s", "%H:%M:%S")
    )
    handler.addFilter(lambda _: handler.flush() or True)
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(prog="zippo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="Collect tests from the given paths"
    )
    collect_parser.add_argument(
        "paths", nargs="+", type=Path, help="Files or directories to collect tests from"
    )
    collect_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=_default_jobs(),
        help="Number of parallel jobs",
    )
    collect_parser.add_argument(
        "-m",
        "--marker",
        default=None,
        help="Only collect tests matching the given marker expression",
    )
    collect_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file for JSON results",
    )

    shard_parser = subparsers.add_parser(
        "shard", help="Split collected tests into shards for parallel execution"
    )
    shard_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input JSON file from the collect command",
    )
    shard_parser.add_argument(
        "-n",
        "--num-shards",
        type=int,
        required=True,
        help="Total number of shards",
    )
    shard_parser.add_argument(
        "-s",
        "--shard-id",
        type=int,
        required=True,
        help="Shard index (0-based)",
    )
    shard_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output file with one test node ID per line",
    )

    markers_parser = subparsers.add_parser(
        "list-markers", help="List pytest markers from the given test files"
    )
    markers_parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to collect markers from",
    )
    markers_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=_default_jobs(),
        help="Number of parallel jobs",
    )
    args = parser.parse_args()
    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "shard":
        cmd_shard(args)
    elif args.command == "list-markers":
        cmd_list_markers(args)


if __name__ == "__main__":
    main()

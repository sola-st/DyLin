import os
import json
from pathlib import Path
from typing import Optional

from fire import Fire


def _dylin_coverage_json_for_reports(reports_dir: Path) -> Optional[Path]:
    """
    Return the merged coverage file for a GitHub project run.

    DynaPyt may write many ``coverage-<uuid>.json`` shards plus a merged ``coverage.json`` under
    ``dynapyt_coverage/dynapyt_coverage-<session>/``. We must pick ``coverage.json``, not the shards.
    """
    session_dirs = sorted(p for p in reports_dir.glob("dynapyt_coverage/dynapyt_coverage-*") if p.is_dir())
    if not session_dirs:
        for legacy in sorted(p for p in reports_dir.glob("dynapyt_coverage-*") if p.is_dir()):
            main = legacy / "coverage.json"
            if main.is_file():
                return main
        return None
    for session in session_dirs:
        main = session / "coverage.json"
        if main.is_file():
            return main
    return None


def _test_cov_json(test_dir: Path, i: int) -> Optional[Path]:
    """pytest-cov may write cov.json at testcov_<i>/cov.json or nested testcov_<i>/testcov/cov.json after a bad mv."""
    p1 = test_dir / f"testcov_{i}" / "cov.json"
    p2 = test_dir / f"testcov_{i}" / "testcov" / "cov.json"
    if p1.is_file():
        return p1
    if p2.is_file():
        return p2
    return None


def _timing_txt_for_coverage_json(coverage_json: Path) -> Path:
    """timing.txt sits under reports_<i>/ next to dynapyt_output/ and dynapyt_coverage/."""
    # .../reports_<i>/dynapyt_coverage/dynapyt_coverage-.../coverage.json -> parents up to reports_<i>
    return coverage_json.parent.parent.parent / "timing.txt"


def _github_project_count() -> int:
    """Number of lines in scripts/projects.txt (GitHub benchmark size)."""
    p = Path(__file__).resolve().parent / "projects.txt"
    if not p.is_file():
        return 37
    with open(p, encoding="utf-8") as f:
        return len([ln for ln in f if ln.strip() and not ln.strip().startswith("#")])

def sanity_check(analysis_coverage, test_coverage):
    X = 0 #len("/opt/dylinVenv/lib/python3.10/site-packages/")
    for file, lines in analysis_coverage.items():
        if file[X:-5] not in test_coverage["files"]:
            print(f"File {file} not in test coverage")
            continue
        for line, analyses in lines.items():
            if int(line) not in test_coverage["files"][file[X:-5]]["executed_lines"]:
                print(f"Line {line} not in test coverage for {file}")
                continue
                

def coverage_report(analysis_coverage: str, test_coverage: str):
    with open(analysis_coverage) as f:
        coverage = json.load(f)
    with open(test_coverage) as f:
        content = json.load(f)
        test_coverage = content["totals"]["covered_lines"]
    
    # sanity_check(coverage, content)
    
    covered_by = {}
    total_covered_lines = 0
    for file, lines in coverage.items():
        if file.startswith("/opt/dylinVenv/lib/python3.10/site-packages/"):
            fl = file[40:-5]
        elif file.startswith("/Work/"):
            fl = file[6:-5]
        else:
            fl = file[:-5]
        if fl not in content["files"] and file[:-5] not in content["files"]:
            continue
        if fl not in content["files"]:
            fl = file[:-5]
        for line, analyses in lines.items():
            if int(line) not in content["files"][fl]["executed_lines"]:
                continue
            total_covered_lines += 1
            for analysis, count in analyses.items():
                if analysis not in covered_by:
                    covered_by[analysis] = 0
                covered_by[analysis] += 1
    return covered_by, total_covered_lines, test_coverage

def compare_only_one(analysis_dir: str, test_dir: str):
    test_cov = Path(test_dir) / "cov.json"
    for ac in Path(analysis_dir).glob("**/dynapyt_coverage-*/coverage*.json"):
        print(f"{ac} {test_cov}")
        covered_by, total_covered_lines, test_coverage = coverage_report(str(ac), str(test_cov))
        with open("coverage_comparison.csv", "a") as f:
                f.write(f"{str(ac)} {total_covered_lines}, {test_coverage}\n")

def coverage_comparison(
    analysis_dir: str,
    test_dir: str,
    max_project: Optional[int] = None,
    out_csv: str = "coverage_comparison.csv",
    strict: bool = False,
):
    """
    Compare DyLin analysis coverage to pytest test coverage per GitHub project index.

    By default loops 1..N where N is the number of projects in ``scripts/projects.txt`` (37).
    Pass ``max_project`` to override (e.g. smoke test on first 3 projects).
    """
    analysis_dir = Path(analysis_dir).resolve()
    test_dir = Path(test_dir).resolve()
    n = max_project if max_project is not None else _github_project_count()
    rows_written = 0
    missing_dylin = 0
    missing_testcov = 0
    missing_timing = 0

    for i in range(1, n + 1):
        reports_dir = analysis_dir / f"reports_{i}"
        if not os.path.exists(reports_dir):
            print(f"Missing reports directory for project index {i}: {reports_dir}")
            missing_dylin += 1
            continue

        analysis_coverage = _dylin_coverage_json_for_reports(reports_dir)
        if analysis_coverage is None:
            print(f"No DyLin coverage.json for project index {i} ({reports_dir})")
            missing_dylin += 1
            continue

        test_coverage = _test_cov_json(test_dir, i)
        if test_coverage is None or not test_coverage.exists():
            print(
                f"Missing test coverage cov.json for project index {i} "
                f"(expected {test_dir}/testcov_{i}/cov.json or .../testcov_{i}/testcov/cov.json)"
            )
            missing_testcov += 1
            continue
        if not analysis_coverage.exists():
            print(f"Missing DyLin coverage file for project index {i}: {analysis_coverage}")
            missing_dylin += 1
            continue
        timing_path = _timing_txt_for_coverage_json(analysis_coverage)
        if not timing_path.exists():
            print(f"Missing timing.txt for project index {i} (expected {timing_path})")
            missing_timing += 1
            continue
        with open(timing_path) as f:
            timing = f.read().strip()
        project_name = timing.split(" ")[0]
        covered_by, total_covered_lines, test_coverage = coverage_report(analysis_coverage, test_coverage)
        with open(out_csv, "a") as f:
            f.write(f"{i}, {total_covered_lines}, {test_coverage}, {project_name}\n")
        rows_written += 1

    print(
        "coverage_comparison summary: "
        f"rows_written={rows_written} "
        f"missing_dylin={missing_dylin} "
        f"missing_testcov={missing_testcov} "
        f"missing_timing={missing_timing}"
    )

    if strict and (missing_dylin + missing_testcov + missing_timing) > 0:
        raise SystemExit(1)
        # print(f"Project {i}:")
        # print(f"Analysis covered lines: {total_covered_lines}")
        # print(f"Test coverage: {test_coverage}")

if __name__ == '__main__':
    Fire()
                
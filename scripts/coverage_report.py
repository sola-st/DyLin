import json
from pathlib import Path
from fire import Fire

def coverage_report(analysis_coverage: str, test_coverage: str):
    with open(analysis_coverage) as f:
        coverage = json.load(f)
    with open(test_coverage) as f:
        test_coverage = json.load(f)["totals"]["covered_lines"]
    
    covered_by = {}
    total_covered_lines = 0
    for file, lines in coverage.items():
        for line, analyses in lines.items():
            total_covered_lines += 1
            for analysis, count in analyses.items():
                if analysis not in covered_by:
                    covered_by[analysis] = 0
                covered_by[analysis] += 1
    return covered_by, total_covered_lines, test_coverage

def coverage_comparison(analysis_dir: str, test_dir: str):
    for i in range(1, 41):
        analysis_coverage = list((Path(analysis_dir)/f"reports_{i}").glob("dynapyt_coverage-*/coverage.json"))
        if len(analysis_coverage) != 1:
            print(f"Problem in coverage for {i}")
            continue
        analysis_coverage = analysis_coverage[0]
        test_coverage = Path(test_dir)/f"testcov_{i}/cov.json"
        if not analysis_coverage.exists() or not test_coverage.exists():
            print(f"Problem in coverage for {i}")
            continue
        covered_by, total_covered_lines, test_coverage = coverage_report(analysis_coverage, test_coverage)
        with open("coverage_comparison.csv", "a") as f:
            f.write(f"{i}, {total_covered_lines}, {test_coverage}\n")
        # print(f"Project {i}:")
        # print(f"Analysis covered lines: {total_covered_lines}")
        # print(f"Test coverage: {test_coverage}")

if __name__ == '__main__':
    Fire(coverage_comparison)
                
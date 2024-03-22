import json
from pathlib import Path
from fire import Fire

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

def coverage_comparison(analysis_dir: str, test_dir: str):
    for i in range(1, 50):
        analysis_coverage = list((Path(analysis_dir).resolve()/f"reports_{i}").glob("dynapyt_coverage-*/coverage.json"))
        if len(analysis_coverage) != 1:
            print(f"There is not 1 file for DyLin coverage for {i} {analysis_coverage}")
            continue
        analysis_coverage = analysis_coverage[0]
        test_coverage = Path(test_dir)/f"testcov_{i}/cov.json"
        if not analysis_coverage.exists() or not test_coverage.exists():
            print(f"One coverage does not exist {i}")
            continue
        covered_by, total_covered_lines, test_coverage = coverage_report(analysis_coverage, test_coverage)
        with open("coverage_comparison.csv", "a") as f:
            f.write(f"{i}, {total_covered_lines}, {test_coverage}\n")
        # print(f"Project {i}:")
        # print(f"Analysis covered lines: {total_covered_lines}")
        # print(f"Test coverage: {test_coverage}")

if __name__ == '__main__':
    Fire(coverage_comparison)
                
import json
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

if __name__ == '__main__':
    Fire(coverage_report)
                
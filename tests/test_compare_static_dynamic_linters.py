import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_static_dynamic_linters import compare, parse_warning_line, read_warnings_from_file


def test_parse_dynamic_code_prefixed_warning(tmp_path: Path):
    line = "F841: foo.py: 10: unused variable"
    warning = parse_warning_line(line, tmp_path)

    assert warning is not None
    assert warning.file.endswith("foo.py")
    assert warning.line == "10"
    assert warning.column is None
    assert warning.raw == line


def test_compare_matches_static_results_and_dynamic_output(tmp_path: Path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    static_file = static_dir / "results_ruff.txt"
    static_file.write_text("foo.py:10:1: F841 local variable 'x' is assigned to but never used\n")

    dynamic_file = tmp_path / "dynamic.txt"
    dynamic_file.write_text("F841: foo.py: 10: unused variable\n")

    matches = compare(str(static_dir), str(dynamic_file))

    assert matches == 1


def test_parse_dynamic_json_warnings(tmp_path: Path):
    output = [
        {
            "results": {
                "InvalidComparisonAnalysis": {
                    "results": {
                        "A-13": [
                            {
                                "finding": {
                                    "location": {
                                        "file": "foo.py",
                                        "start_line": 10,
                                        "start_column": 1,
                                    },
                                    "msg": "compared x to None",
                                }
                            }
                        ]
                    }
                }
            }
        }
    ]
    json_file = tmp_path / "output.json"
    json_file.write_text(json.dumps(output))

    warnings = read_warnings_from_file(json_file, tmp_path, dynamic=True)
    assert len(warnings) == 1
    assert warnings[0].line == "10"
    assert warnings[0].file.endswith("foo.py")

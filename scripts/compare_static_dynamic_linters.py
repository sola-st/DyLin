import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fire

WARNING_RE = re.compile(r'^(?P<file>.+?):(?P<line>\d+)(?::(?P<column>\d+))?:\s*(?P<rest>.*)$')
CODE_FILE_LINE_RE = re.compile(r'^(?P<code>[A-Z][A-Z0-9_-]*):\s*(?P<file>.+?):\s*(?P<line>\d+):\s*(?P<rest>.*)$')


@dataclass(frozen=True)
class Warning:
    file: str
    line: str
    column: Optional[str]
    raw: str
    source: str
    source_path: str

    @property
    def line_key(self) -> str:
        return f"{self.file}:{self.line}"

    @property
    def column_key(self) -> str:
        if self.column:
            return f"{self.file}:{self.line}:{self.column}"
        return self.line_key

    @property
    def basename_key(self) -> str:
        return f"{Path(self.file).name}:{self.line}"

    @property
    def suffix_key(self) -> str:
        parts = Path(self.file).parts
        suffix = "/".join(parts[-3:]) if len(parts) >= 3 else self.file
        return f"{suffix}:{self.line}"


def normalize_path(path_str: str, base: Path) -> str:
    path = Path(path_str.strip())
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path.as_posix()


def parse_warning_line(line: str, base: Path) -> Optional[Warning]:
    text = line.strip()
    if not text or text.startswith("#"):
        return None

    match = CODE_FILE_LINE_RE.match(text)
    if match:
        file_path = normalize_path(match.group("file"), base)
        return Warning(
            file=file_path,
            line=match.group("line"),
            column=None,
            raw=text,
            source="dynamic",
            source_path=str(base),
        )

    match = WARNING_RE.match(text)
    if not match:
        return None

    file_path = normalize_path(match.group("file"), base)
    return Warning(
        file=file_path,
        line=match.group("line"),
        column=match.group("column"),
        raw=text,
        source="unknown",
        source_path=str(base),
    )


def parse_dynamic_json(path: Path, base: Path) -> List[Warning]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    warnings: List[Warning] = []
    if not isinstance(data, list):
        return warnings

    for finding in data:
        if not isinstance(finding, dict):
            continue
        results = finding.get("results")
        if not isinstance(results, dict):
            continue
        for checker in results.values():
            if not isinstance(checker, dict):
                continue
            issue_results = checker.get("results")
            if not isinstance(issue_results, dict):
                continue
            for issue_list in issue_results.values():
                if not isinstance(issue_list, list):
                    continue
                for issue in issue_list:
                    fnd = issue.get("finding")
                    if not isinstance(fnd, dict):
                        continue
                    location = fnd.get("location")
                    if not isinstance(location, dict):
                        continue
                    file_path = location.get("file")
                    line = location.get("start_line")
                    if not file_path or not isinstance(line, int):
                        continue
                    warnings.append(
                        Warning(
                            file=normalize_path(file_path, base),
                            line=str(line),
                            column=None,
                            raw=json.dumps(issue),
                            source="dynamic_json",
                            source_path=str(path),
                        )
                    )
    return warnings


def resolve_static_files(static_dir: Path) -> List[Path]:
    if static_dir.is_file():
        return [static_dir]
    patterns = ["**/results_*.txt", "**/results.txt"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(static_dir.glob(pattern)))
    return files


def resolve_dynamic_files(dynamic_path: Path) -> List[Path]:
    if dynamic_path.is_file():
        return [dynamic_path]
    patterns = ["**/DyLin_findings.txt", "**/output.txt", "**/*.txt", "**/*.json"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(dynamic_path.glob(pattern)))
    return files


def read_warnings_from_file(path: Path, base: Path, dynamic: bool = False) -> List[Warning]:
    if dynamic and path.suffix.lower() == ".json":
        return parse_dynamic_json(path, base)

    warnings: List[Warning] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_warning_line(line, base)
            if parsed:
                warnings.append(parsed)
    return warnings


def collect_warnings(static_dir: Path, dynamic_path: Path) -> Tuple[List[Warning], List[Warning]]:
    static_files = resolve_static_files(static_dir)
    if not static_files:
        raise FileNotFoundError(f"No static warning files found under {static_dir}")

    dynamic_files = resolve_dynamic_files(dynamic_path)
    if not dynamic_files:
        raise FileNotFoundError(f"No dynamic warning files found under {dynamic_path}")

    static_warnings: List[Warning] = []
    for file_path in static_files:
        static_warnings.extend(read_warnings_from_file(file_path, file_path.parent, dynamic=False))

    dynamic_warnings: List[Warning] = []
    for file_path in dynamic_files:
        dynamic_warnings.extend(read_warnings_from_file(file_path, file_path.parent, dynamic=True))

    return static_warnings, dynamic_warnings


def match_warnings(static_warnings: Iterable[Warning], dynamic_warnings: Iterable[Warning]) -> List[Tuple[Warning, Warning]]:
    index: Dict[str, List[Warning]] = defaultdict(list)
    for warning in static_warnings:
        index[warning.line_key].append(warning)
        index[warning.column_key].append(warning)
        index[warning.basename_key].append(warning)
        index[warning.suffix_key].append(warning)

    matches: List[Tuple[Warning, Warning]] = []
    seen: set[Tuple[str, str]] = set()
    for warning in dynamic_warnings:
        for key in [warning.line_key, warning.column_key, warning.basename_key, warning.suffix_key]:
            for static_warning in index.get(key, []):
                pair = (static_warning.raw, warning.raw)
                if pair not in seen:
                    matches.append((static_warning, warning))
                    seen.add(pair)
    return matches


def compare(static_dir: str, dynamic: str):
    static_dir_path = Path(static_dir)
    dynamic_path = Path(dynamic)
    static_warnings, dynamic_warnings = collect_warnings(static_dir_path, dynamic_path)

    matches = match_warnings(static_warnings, dynamic_warnings)
    for static_warning, dynamic_warning in matches:
        print(f"Static: {static_warning.raw} Dynamic: {dynamic_warning.raw}")

    print(f"Found {len(static_warnings)} static warnings, {len(dynamic_warnings)} dynamic warnings, {len(matches)} matches.")
    return len(matches)


if __name__ == "__main__":
    Fire(compare)

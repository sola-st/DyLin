from fire import Fire


def check_findings(ruff_res: str):
    with open(ruff_res, "r") as file:
        ruff_findings = file.read().split("\n")
    for fnd in ruff_findings:
        if not fnd.startswith("tests"):
            continue
        parts = fnd.split(":")
        file_path = parts[0]
        line = int(parts[1])
        with open(file_path, "r") as file:
            lines = file.read().split("\n")
        line_content = lines[line - 1]
        if "# DyLin warn" in line_content:
            print(fnd)
            print(line_content)


if __name__ == "__main__":
    Fire(check_findings)

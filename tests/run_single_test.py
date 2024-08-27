from os import sep, remove, environ
from os.path import join, exists
from shutil import copyfile, move, rmtree
from typing import Tuple
import pytest
import json
import yaml
from pathlib import Path

from dynapyt.instrument.instrument import instrument_file
from dynapyt.utils.hooks import get_hooks_from_analysis
from dynapyt.run_analysis import run_analysis
from dynapyt.utils.runtimeUtils import gather_output


def test_runner(directory_pair: Tuple[str, str], capsys):
    abs_dir, rel_dir = directory_pair

    # load warnings
    with open(join(abs_dir, "program.py"), "r") as file:
        lines = file.read().splitlines()
    expected_warnings = []
    for ln, line in enumerate(lines):
        if "# DyLin warn" in line:
            expected_warnings.append((line.split("# DyLin warn")[0].strip(), ln + 1))

    # load checkers
    checkers_file = join(abs_dir, "checkers.txt")
    with open(checkers_file, "r") as file:
        checkers = file.read().splitlines()

    analysis_names = []
    for i, checker in enumerate(checkers):
        if ";" in checker:
            parts = checker.split(";")
            the_analysis = parts[0]
            analysis_name = the_analysis.split(".")[-1]
            for j in range(1, len(parts)):
                if parts[j].startswith("config="):
                    the_config = parts[j].split("=")[1]
                    config_path = Path(the_config).resolve()
                    parts[j] = f"config={config_path}"
                    with open(config_path, "r") as yaml_str:
                        yml = yaml.safe_load(yaml_str)

                    name = yml.get("name")

                    if name:
                        analysis_name = name
            checkers[i] = ";".join(parts)
            analysis_names.append(analysis_name)
        else:
            analysis_names.append(checker.split(".")[-1])

    # gather hooks used by the analysis
    selected_hooks = get_hooks_from_analysis(checkers)

    # instrument
    program_file = join(abs_dir, "program.py")
    orig_program_file = join(abs_dir, "program.py.orig")
    # make sure to instrument the uninstrumented version
    run_as_file = False
    with open(program_file, "r") as file:
        src = file.read()
        if "DYNAPYT: DO NOT INSTRUMENT" in src:
            if not exists(orig_program_file):
                pytest.fail(f"Could find only the instrumented program in {rel_dir}")
            copyfile(orig_program_file, program_file)

    instrument_file(program_file, selected_hooks)

    if exists(join(abs_dir, "__init__.py")) and not exists(join(abs_dir, "__init__.py.orig")):
        instrument_file(join(abs_dir, "__init__.py"), selected_hooks)

    # analyze

    captured = capsys.readouterr()
    print(captured.out)
    exception_thrown = None
    try:
        session_id = run_analysis(
            f"{rel_dir.replace(sep, '.')}.program",
            checkers,
            output_dir=abs_dir,
        )
    except Exception as e:
        exception_thrown = e
        session_id = environ.get("DYNAPYT_SESSION_ID", None)

    captured = capsys.readouterr()
    print(captured.out)

    # check output
    fail = []
    gather_output(Path(abs_dir) / f"dynapyt_output-{session_id}")
    for analysis_name in analysis_names:
        with open(join(abs_dir, f"dynapyt_output-{session_id}", f"output.json"), "r") as file:
            analysis_output = json.load(file)[0]
        # print(analysis_output)
        for wcode, findings in analysis_output["results"][0][analysis_name]["results"].items():
            for finding in findings:
                if finding["finding"]["location"]["start_line"] not in [ew[1] for ew in expected_warnings]:
                    fail.append(f"Found something weird: {finding}")
        for expected_warning in expected_warnings:
            found = False
            for wcode, findings in analysis_output["results"][0][analysis_name]["results"].items():
                for finding in findings:
                    if finding["finding"]["location"]["start_line"] == expected_warning[1]:
                        found = True
            if not found:
                fail.append(f"Expected warning not found: {expected_warning}")

    # restore uninstrumented program and remove temporary files
    move(orig_program_file, program_file)
    remove(join(abs_dir, "program-dynapyt.json"))
    remove(join(abs_dir, f"dynapyt_output-{session_id}", "findings.csv"))
    remove(join(abs_dir, f"dynapyt_output-{session_id}", "findings.csv.lock"))
    for analysis_name in analysis_names:
        if not fail:
            remove(join(abs_dir, f"dynapyt_output-{session_id}", f"output-{analysis_name}report.json.lock"))
            remove(join(abs_dir, f"dynapyt_output-{session_id}", f"output.json"))
    if exists(join(abs_dir, "__init__.py")) and exists(join(abs_dir, "__init__.py.orig")):
        move(join(abs_dir, "__init__.py.orig"), join(abs_dir, "__init__.py"))
        remove(join(abs_dir, "__init__-dynapyt.json"))
    rmtree(join(abs_dir, "__pycache__"))
    if not fail:
        rmtree(join(abs_dir, f"dynapyt_output-{session_id}"))

    if fail:
        pytest.fail("\n".join(fail))
    if exception_thrown is not None:
        pytest.fail(str(exception_thrown))
    # for failure in fail:
    #     pytest.fail(failure)

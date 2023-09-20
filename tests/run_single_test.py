from importlib import import_module
from os import sep, remove
from os.path import join, exists
from shutil import copyfile, move, rmtree
from typing import Tuple
import pytest
import json

from dynapyt.instrument.instrument import instrument_file
from dynapyt.utils.hooks import get_hooks_from_analysis
from dynapyt.run_analysis import run_analysis


def test_runner(directory_pair: Tuple[str, str], capsys):
    import dynapyt.runtime as _rt

    abs_dir, rel_dir = directory_pair

    # load warnings
    with open(join(abs_dir, "program.py"), "r") as file:
        lines = file.read().splitlines()
    expected_warnings = []
    for ln, line in enumerate(lines):
        if "# DyLin warn" in line:
            expected_warnings.append((line.split("# DyLin warn")[1].strip(), ln + 1))

    # load checkers
    checkers_file = join(abs_dir, "checkers.txt")
    with open(checkers_file, "r") as file:
        checkers = file.read().splitlines()

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
    analysis_instances = []
    analysis_names = []
    for analysis in checkers:
        if ":" not in analysis:
            analysis_path = analysis
            config = None
        else:
            analysis_path, config = analysis.split(":")
        module_prefix, analysis_name = analysis_path.rsplit(".", 1)
        analysis_names.append(analysis_name)
        module = import_module(module_prefix)
        analysis_class = getattr(module, analysis_name)
        analysis_instances.append(analysis_class(config=config, report_path=abs_dir))

    _rt.set_analysis(analysis_instances)

    captured = capsys.readouterr()
    for analysis_instance in analysis_instances:
        if hasattr(analysis_instance, "begin_execution"):
            analysis_instance.begin_execution()
    import_module(f"{rel_dir.replace(sep, '.')}.program")
    _rt.end_execution()
    del _rt
    del analysis_instances

    captured = capsys.readouterr()
    print(captured.out)

    # check output
    fail = []
    for analysis_name in analysis_names:
        with open(join(abs_dir, f"{analysis_name}report.json"), "r") as file:
            analysis_output = json.load(file)
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
    remove(join(abs_dir, "findings.csv"))
    remove(join(abs_dir, "findings.csv.lock"))
    for analysis_name in analysis_names:
        remove(join(abs_dir, f"{analysis_name}report.json"))
        remove(join(abs_dir, f"{analysis_name}report.json.lock"))
    if exists(join(abs_dir, "__init__.py")) and exists(join(abs_dir, "__init__.py.orig")):
        move(join(abs_dir, "__init__.py.orig"), join(abs_dir, "__init__.py"))
        remove(join(abs_dir, "__init__-dynapyt.json"))
    rmtree(join(abs_dir, "__pycache__"))

    for failure in fail:
        pytest.fail(failure)

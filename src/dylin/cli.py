import argparse
import sys
import tempfile
from pathlib import Path

import docker

from dylin.select_checkers import select_checkers

# Repo root: src/dylin/cli.py -> src/dylin -> src -> repo root
_DYLIN_ROOT = Path(__file__).parent.parent.parent.resolve()


def _build_entrypoint_script(tmp_output_dir, setup_cmd, run_command, coverage, quiet):
    coverage_env = f'export DYNAPYT_COVERAGE="{tmp_output_dir}"\n' if coverage else ""
    coverage_arg = f'--coverage_dir={tmp_output_dir}' if coverage else '--coverage_dir=""'
    run_output_redirect = "" if quiet else f" > {tmp_output_dir}/run_output.log 2>&1"

    return f"""\
#!/bin/bash
export PYTHONUNBUFFERED=1
cp -r /project_root /tmp/project
cd /tmp/project
{setup_cmd}
export PYTHONPATH="/analysis:$PYTHONPATH"
python -m dynapyt.run_instrumentation --directory . --analysisFile /analysis/final_analysis.txt > {tmp_output_dir}/dynapyt_instrumentation.log 2>&1 || exit 1
export DYNAPYT_SESSION_ID="1234-abcd"
{coverage_env}cp /analysis/final_analysis.txt /tmp/dynapyt_analyses-1234-abcd.txt
{run_command}{run_output_redirect}
python -m dynapyt.post_run {coverage_arg} --output_dir={tmp_output_dir} > {tmp_output_dir}/dynapyt_post_run.log 2>&1
if [ -f "{tmp_output_dir}/output.json" ]; then
    python -m dylin.format_output --findings_path {tmp_output_dir}/output.json > {tmp_output_dir}/output.txt
fi
"""


def instrument_and_run_analysis(
    project_root, analysis_file, output_dir, setup_cmd, run_command, coverage=False, quiet=False
):
    """Docker-based DynaPyt instrumentation + run, with DyLin pre-installed in the container."""
    client = docker.from_env(timeout=240)

    # Build a Docker image with DynaPyt (DyLin is installed at runtime from mounted source)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        dockerfile_content = """\
FROM python:3.13-slim
RUN apt-get update && apt-get install -y gcc git
RUN pip install --upgrade pip
RUN pip install git+https://github.com/sola-st/DynaPyt.git@main#egg=dynapyt
RUN pip install git+https://github.com/sola-st/DyLin.git@main#egg=dylin
"""
        (temp_dir_path / "Dockerfile").write_text(dockerfile_content)

        print("Building Docker image with DynaPyt installed...")
        try:
            image, logs = client.images.build(path=str(temp_dir_path), tag="dynapyt_runner", rm=True)
            for log in logs:
                if "stream" in log:
                    print(log["stream"], end="")
        except docker.errors.BuildError as e:
            print("Failed to build Docker image:")
            for log in e.build_logs:
                if "stream" in log:
                    print(log["stream"], end="")
            return

    print(f"\nRunning container for project {project_root}...")
    print(f"Analysis file: {analysis_file}")
    print(f"Setup command: {setup_cmd}")
    print(f"Output directory: {output_dir}")
    print(f"Run command: {run_command}")

    tmp_output_dir = "/tmp/dynapyt_output"

    # Rewrite analysis file so output_dir points to the container path
    final_analysis_file = analysis_file.parent / "final_analysis.txt"
    content = analysis_file.read_text()
    with open(final_analysis_file, "w") as f:
        for line in content.splitlines():
            if ";output_dir=" not in line:
                f.write(line + f";output_dir={tmp_output_dir}\n")
            else:
                analysis, _ = line.split(";output_dir=")
                f.write(analysis + f";output_dir={tmp_output_dir}\n")

    entrypoint_script = _build_entrypoint_script(
        tmp_output_dir=tmp_output_dir,
        setup_cmd=setup_cmd,
        run_command=run_command,
        coverage=coverage,
        quiet=quiet,
    )

    try:
        container = client.containers.run(
            "dynapyt_runner",
            command=["/bin/bash", "-c", f"{entrypoint_script}"],
            mounts=[
                docker.types.Mount(target="/project_root", source=str(project_root), type="bind", read_only=True),
                docker.types.Mount(target="/analysis", source=str(analysis_file.parent), type="bind", read_only=False),
                docker.types.Mount(target="/tmp/dynapyt_output", source=str(output_dir), type="bind", read_only=False),
                docker.types.Mount(target="/dylin_src", source=str(_DYLIN_ROOT), type="bind", read_only=True),
            ],
            remove=True,
            detach=True,
            tty=False,
        )
    except docker.errors.ContainerError as e:
        print(f"Container error: {e}")
        try:
            print(e.container.logs().decode("utf-8"))
        except docker.errors.NotFound:
            pass

    try:
        # Capture the exit code
        result = container.wait()
        exit_code = result.get("StatusCode", 0)
        print(f"\nContainer finished with exit code: {exit_code}")
    except docker.errors.NotFound:
        pass

    try:
        container.remove(force=True)
    except docker.errors.NotFound:
        pass
    except docker.errors.APIError as e:
        print(f"Failed to remove container: {e}")


def main():
    parser = argparse.ArgumentParser(description="DyLin: Dynamic Linter for Python")
    parser.add_argument(
        "--project-root",
        required=True,
        help="Path to the root directory of the project to analyze.",
    )
    parser.add_argument(
        "--include",
        default="all",
        help='Comma-separated list of checkers to include (default: "all").',
    )
    parser.add_argument(
        "--exclude",
        default="none",
        help='Comma-separated list of checkers to exclude (default: "none").',
    )
    parser.add_argument(
        "--analysis",
        default=None,
        help="Path to an analysis file. If provided, overrides --include/--exclude.",
    )
    parser.add_argument(
        "--setup",
        required=False,
        default="",
        help="Setup script/command to run inside the container before instrumentation.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to save the raw output from DyLin/DynaPyt.",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Store coverage data in the same directory as the analysis output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run the target command without writing run_output.log.",
    )
    parser.add_argument(
        "run_command",
        nargs=argparse.REMAINDER,
        help="Command to run after instrumentation (e.g. `pytest tests` or `python main.py`).",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists() or not project_root.is_dir():
        print(
            f"Error: Project root '{project_root}' does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cmd_list = args.run_command
    if run_cmd_list and run_cmd_list[0] == "--":
        run_cmd_list = run_cmd_list[1:]
    if not run_cmd_list:
        print("Error: No run command provided.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    run_command = " ".join(run_cmd_list)

    # Build the checkers/analysis list
    if args.analysis is not None:
        analysis_file = Path(args.analysis).resolve()
    else:
        checkers_str = select_checkers(include=args.include, exclude=args.exclude)
        if not checkers_str.strip():
            print(
                "Error: No checkers selected with the given --include/--exclude parameters.",
                file=sys.stderr,
            )
            sys.exit(1)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="dylin_analysis_", delete=False)
        tmp.write(checkers_str)
        tmp.close()
        analysis_file = Path(tmp.name)

    instrument_and_run_analysis(
        project_root=project_root,
        analysis_file=analysis_file,
        output_dir=output_dir,
        setup_cmd=args.setup,
        run_command=run_command,
        coverage=args.coverage,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

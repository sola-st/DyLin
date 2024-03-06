import argparse
import datetime
from genericpath import isfile
from multiprocessing import Pool
import os
import pathlib
from posixpath import join
import subprocess
import time
from typing import List
import json

from dynapyt.run_instrumentation import instrument_dir
from dynapyt.run_analysis import run_analysis

from pebble import ProcessExpired, ProcessPool

parser = argparse.ArgumentParser()
parser.add_argument("--number", help="Number of submissions to prepare")
parser.add_argument("--competition", help="Selects competition")
parser.add_argument("--path", help="Directory to work in")
parser.add_argument("--only-run", help="Only run analysis", default=False, action="store_true")
parser.add_argument("--only-prepare", help="Only prepare files to run", default=False, action="store_true")
parser.add_argument("--search", help="Search query")
parser.add_argument("--kaggleConf", help="Kaggle config dir")
args = parser.parse_args()

os.environ["KAGGLE_CONFIG_DIR"] = args.kaggleConf

from kaggle.models.kaggle_models_extended import Kernel

"""
For some competitions the rules have to be accepted on kaggles website first!
Also this script requires an active python environment which has jupyter, DynaPyt and dylin installed

Steps:
1. Select competition & number
2. Download number of notebooks
3. Download input files to ../input and /kaggle/input
4. Convert them to *.py
5. Remove all iphython() calls
7. Instrument files
8. Run all analyses in parallel
"""

path = pathlib.Path(args.path)

if not args.only_run:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.api_client import ApiClient

    api = KaggleApi(ApiClient())
    api.authenticate()
    print("authenticated")

    """
    STEP 1: Select competition and number of submissions to download
    """
    nmb_submissions = int(args.number)
    competition = args.competition if not args.competition is None else "titanic"

    """
    STEP 2: Download number of notebooks

    Assumes kaggle api is installed
    """

    def dl_kernels(page, page_size):
        print("searching")
        return api.kernels_list(page=page, page_size=page_size, competition=competition, search=args.search)

    to_download = nmb_submissions
    page = 1

    kernels: List[Kernel] = []
    while to_download >= 1:
        tmp_kernels = []
        # max page size is 100
        if to_download >= 100:
            tmp_kernels = dl_kernels(page, 100)
        else:
            tmp_kernels = dl_kernels(page, to_download)

        if len(tmp_kernels) == 0:
            print("Can't find more kernels matching criteria")
            break

        # kaggle does not provide us with a score, therefore we filter by number of votes
        # Note: this might skew how representative the sample is but will hopefully filter
        #       out most empty and bad kernels (e.g. they crash right away or don't do anything related to the task)
        filtered = [kernel for kernel in tmp_kernels if kernel.totalVotes > 2 and kernel.isPrivate is False]
        kernels = kernels + filtered
        page = page + 1
        to_download = to_download - len(filtered)

    print(f"found {len(kernels)} competitions to download")

    kernel_infos = {}
    for kernel in kernels:
        try:
            api.kernels_pull(kernel.ref, path)
            print("downloaded " + kernel.ref)
            kernel_infos[kernel.ref] = kernel.__dict__
        except Exception as e:
            print(e)

    """
    STEP 2a: save metadata of each notebook

    Contains number of votes, author name etc.
    """

    class DateTimeEncoder(json.JSONEncoder):
        def default(self, z):
            if isinstance(z, datetime.datetime):
                return str(z)
            else:
                return super().default(z)

    with open(path / "info.json", "a+") as report:
        report.write(json.dumps(kernel_infos, indent=4, cls=DateTimeEncoder))

    print("downloaded submissions")

    """
    STEP 3: download competitions datasets

    We download to ../input /kaggle/input and /kaggle/input/<competitionname> as users can use both paths on the platform
    """

    def dl_datasets(path: pathlib.Path):
        try:
            print("creating dir " + str(path))
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(e)
        api.competition_download_files(competition=competition, path=path)

        # unzip all files and remove zips afterwards
        subprocess.run(f"unzip -o {path}/*.zip -d {path}", shell=True)
        subprocess.run(f"rm {path}/*.zip", shell=True)

    dl_datasets(pathlib.Path("input"))
    dl_datasets(pathlib.Path("input/" + competition))
    dl_datasets(pathlib.Path("/kaggle/input"))
    dl_datasets(pathlib.Path("/kaggle/input/" + competition))

    print("downloaded datasets")

    """
    STEP 4: convert notebooks to *.py files

    Assumes jupyter notebook is installed
    """
    subprocess.run(f"jupyter nbconvert --to script {path}/*.ipynb", shell=True)
    subprocess.run(f"rm {path}/*.ipynb", shell=True)

    print("converted to *.py files")

    """
    STEP 5: comment out all get_ipython() calls

    We consider get_ipython calls to be dangerous as they may install packages etc.
    """
    subprocess.run(f"find {path} -type f -name '*.py' -print0 | xargs -0 sed -i 's/^get_ipython/#&/'", shell=True)

    print("commented out all get_ipython() calls")

    """
    STEP 6: instrument files

    Assumes dynapyt and dylin are installed in currently used pip executable
    """

    # run instrumentation
    here = pathlib.Path(__file__).parent.resolve()
    with open(here / ".." / "dylin_config_kaggle.txt", "r") as f:
        config_content = f.read()
    analyses = config_content.strip().split("\n")
    instrument_dir(path, analyses)
    # subprocess.run(
    #     f"python -m dynapyt.run_instrumentation --directory {path} --module dylin --analysis AnalysisWrapper", shell=True)
    print("finished instrumentation")

if not args.only_prepare:
    """
    STEP 7: run analysis in parallel

    We use pebbles Process Pool here because the default Python Pool does not properly kill processes
    """

    def run_dylin(path):
        run_analysis(path, analyses, coverage=True)
        # result = subprocess.run(f"python -m dynapyt.run_analysis --entry {path} --analysis AnalysisWrapper --module dylin",
        #                         shell=True)
        # if result.returncode != 0:
        #     print(f"Error at {result}")
        # else:
        #     print(f"done with {result}")

    onlypy = [join(path, f) for f in os.listdir(path) if isfile(join(path, f)) and f.endswith("py")]

    start_time = time.time()
    print("#################### starting analyses...")

    TIMEOUT_SECONDS = 5 * 60

    nmb_timeouts = 0
    nmb_errors = 0

    # We explicitly allow workers to only work on 1 task to prevent memory leaks and to limit memory fragmentation
    with ProcessPool(max_workers=10, max_tasks=1) as pool:
        future = pool.map(run_dylin, onlypy, timeout=TIMEOUT_SECONDS)

        iterator = future.result()

        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError as error:
                print(f"run took longer than {TIMEOUT_SECONDS} seconds")
                nmb_timeouts = nmb_timeouts + 1
            except ProcessExpired as error:
                print(f"{error} Exit code: {error.exitcode}")
                if error.exitcode != 0:
                    nmb_errors = nmb_errors + 1
            except Exception as error:
                print("run raised %s" % error)
                if "timeout" in str(error):
                    nmb_timeouts = nmb_timeouts + 1
                else:
                    nmb_errors = nmb_errors + 1

    print(f"#################### done - took {time.time() - start_time}")
    print(f"#################### number errors: {nmb_errors} number timeouts: {nmb_timeouts}")

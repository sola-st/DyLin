from fire import Fire
from pathlib import Path


def project_names():
    here = Path(__file__).parent
    with open(str(here / "projects.txt")) as f:
        projects = f.read().split("\n")
    for project in projects[:35]:
        print(project.split(" ")[0])


if __name__ == "__main__":
    Fire(project_names)

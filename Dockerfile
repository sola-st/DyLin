FROM ubuntu:22.04

WORKDIR /Work

RUN apt update
RUN apt install -q -y python3 python3-pip python3-venv
RUN python3 -m venv /opt/dylinVenv
ENV PATH="/opt/dylinVenv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN apt install -q -y git

RUN mkdir ./reports
RUN chmod -R 777 ./reports

RUN mkdir ./DyLin

COPY ./requirements.txt ./DyLin/requirements.txt
RUN pip install --no-cache-dir -r ./DyLin/requirements.txt

COPY ./scripts ./DyLin/scripts
COPY ./src ./DyLin/src
COPY ./tests ./DyLin/tests
COPY ./pyproject.toml ./DyLin/pyproject.toml
COPY ./README.md ./DyLin/README.md

RUN chmod -R 777 ./DyLin

RUN pip install ./DyLin/

ENTRYPOINT [ "python", "./DyLin/scripts/analyze_repo.py", "--repo" ]
CMD [ "2" ]

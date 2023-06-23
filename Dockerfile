FROM ubuntu:22.04

WORKDIR /DyLin

RUN apt update
RUN apt install -q -y python3 python3-pip python3-venv
RUN python3 -m venv /opt/dylinVenv
ENV PATH="/opt/dylinVenv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN apt install -q -y git

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY dylin/ ./dylin

RUN chmod -R 777 ./dylin

CMD python dylin/scripts/analyze_repo.py --repo 1
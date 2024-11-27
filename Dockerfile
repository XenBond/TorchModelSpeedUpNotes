FROM nvidia/cuda:12.2.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/user/local/bin:${PATH}"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget \
    curl \
    ca-certificates \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN echo "Python installed at: $(which python3)"

RUN python3.10 -m pip --version

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --config python3

RUN python3 --version && pip --version

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

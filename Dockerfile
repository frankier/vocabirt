FROM debian:bullseye-slim

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get dist-upgrade -yq && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -yq \
    zstd \
    xz-utils \
    git \
    build-essential \
    wget \
    bash \
    coreutils \
    bzip2 \
    ca-certificates \
    curl \
    cmake \
    python3 \
    python3-pip \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /vocabirt/

ENV LANG='C.UTF-8' LC_ALL='C.UTF-8'

RUN python3 -m pip install --upgrade poetry==1.1.7

ADD pyproject.toml poetry.lock /vocabirt/

RUN poetry export \
      --without-hashes > requirements.txt && \
    sed -i '/pytorch/d' requirements.txt && \
    python3 -m pip install -r requirements.txt && \
    rm requirements.txt && \
    pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    rm -rf /root/.cache

RUN echo "/vocabirt" > \
    /usr/local/lib/python3.8/dist-packages/vocabirt.pth

RUN ln -sf /usr/bin/python3 /usr/bin/python

ADD . /vocabirt/

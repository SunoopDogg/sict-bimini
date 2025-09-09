FROM python:3.13.7-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive

# curl and ca-certificates dependencies
RUN apt-get update \
    && apt-get install -y \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# git and git-lfs dependencies
RUN apt-get update \
    && apt-get install -y \
        git \
        git-lfs \
    && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# GUI / Rendering dependencies
RUN apt-get update \
    && apt-get install -y \
        libgtk2.0-dev \
        libgl1 \
        tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
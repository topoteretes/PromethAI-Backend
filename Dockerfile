##FROM python:3.10.10-slim
#
#FROM ubuntu:latest
#
#
## The segment starting bellow is important in order to have the chromadb running
#
## Install necessary tools
#RUN apt-get update && \
#    apt-get install -y software-properties-common && \
#    add-apt-repository ppa:ubuntu-toolchain-r/test && \
#    apt-get update && \
#    apt-get install -y gcc-11 g++-11 && \
#    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11 && \
#    apt-get install -y python3-pip && \
#    pip3 install --upgrade pip && \
#    pip3 install --upgrade setuptools wheel && \
#    apt-get install -y build-essential
#RUN apt-get update && apt-get install -y curl
#
## Download and install Miniconda
#RUN if [ "$(uname -m)" = "x86_64" ]; then \
#        curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"; \
#    elif [ "$(uname -m)" = "aarch64" ]; then \
#        curl -sL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh" > "Miniconda3.sh"; \
#    fi && \
#    bash Miniconda3.sh -b -p /miniconda && \
#    rm Miniconda3.sh
## Set path to conda
#ENV PATH=/miniconda/bin:${PATH}
#
## Update conda, install wget, create and activate conda environment "textgen"
#RUN conda update -y conda && \
#    conda install -y wget && \
#    conda create -n textgen python=3.10.9
## Update the package list and install essential tools
#
#
#
## Install CUDA libraries
#RUN /bin/bash -c "source activate textgen && pip install torch torchvision torchaudio"
#
## Add PPA for gcc-11, update packages, install gcc-11, g++-11, update pip and setuptools, install build-essential
#RUN apt-get update && \
#    apt-get install -y gcc-11 g++-11 && \
#    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11 && \
#    pip install --upgrade pip && \
#    pip install --upgrade setuptools wheel && \
#    apt-get install -y build-essential
#
## The segment above this point is important in order to have the chromadb running
#
## Set build argument and environment variable for stage
#ARG STAGE
#ENV STAGE=${STAGE} \
#    PIP_NO_CACHE_DIR=true \
#    PATH="${PATH}:/root/.poetry/bin"
#RUN pip install poetry
## Install Poetry dependencies
#
#
#WORKDIR /app
#COPY pyproject.toml poetry.lock /app/
#
## Install the dependencies
#RUN poetry config virtualenvs.create false && \
#    poetry install --no-root --no-dev
#
#RUN apt-get update -q && \
#    apt-get install curl zip jq netcat-traditional -y -q && \
#    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
#    unzip -qq awscliv2.zip && ./aws/install && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
#
#
##RUN playwright install
##RUN playwright install-deps
#
#WORKDIR /app
#COPY . /app
#COPY entrypoint.sh /app/entrypoint.sh
#RUN chmod +x /app/entrypoint.sh
#
#ENTRYPOINT ["/app/entrypoint.sh"]
FROM python:3.11-slim

# Set build argument
ARG API_ENABLED

# Set environment variable based on the build argument
ENV API_ENABLED=${API_ENABLED} \
    PIP_NO_CACHE_DIR=true
ENV PATH="${PATH}:/root/.poetry/bin"
RUN pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock /app/

# Install the dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev

RUN apt-get update -q && \
    apt-get install curl zip jq netcat-traditional -y -q
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -qq awscliv2.zip && ./aws/install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


#RUN playwright install
#RUN playwright install-deps

WORKDIR /app
COPY . /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
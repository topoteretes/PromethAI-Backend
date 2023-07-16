FROM ubuntu:latest

# Install necessary tools
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11 && \
    apt-get install -y python3.10 python3-pip python3.10-venv && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade setuptools wheel && \
    apt-get install -y build-essential curl
RUN pip install gpt4all
# Create a Python virtual environment
RUN python3.10 -m venv /textgen

# Activate the virtual environment
ENV PATH="/textgen/bin:$PATH"

# Install CUDA libraries
RUN pip install torch torchvision torchaudio embedchain

# Install Poetry dependencies
ARG STAGE
ENV STAGE=${STAGE} \
    PIP_NO_CACHE_DIR=true \
    PATH="${PATH}:/root/.poetry/bin"

RUN pip install poetry
RUN echo $PATH

WORKDIR /app
COPY pyproject.toml poetry.lock /app/

# Install the dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-dev

RUN apt-get update -q && \
    apt-get install curl zip jq netcat-traditional -y -q && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -qq awscliv2.zip && ./aws/install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app
COPY . /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

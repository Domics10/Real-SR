FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 108
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

#package di base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    openssl \
    cmake \
    unzip
#python
RUN apt-get install -y --no-install-recommends \
    python3 &&\
    python3 --version
#altro
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    python3-dev \
    python3-setuptools \
    python3-tk \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    openmpi-bin && \
    rm -rf /var/lib/apt/lists/*

#Forzo l'installazione di pip perché altrimenti prende una versione broken
RUN curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py
RUN python3 get-pip.py --force-reinstall

#Upgrade di pip
RUN apt-get update
RUN pip3 install --upgrade pip
RUN pip3 --version
#Installazione di package tramite pip3
RUN pip3 install termcolor==1.1.0 && pip3 install --upgrade tensorflow-gpu==1.14.0
COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV TORCH_HOME /home/Tencent
WORKDIR $TORCH_HOME
ENV MODE=""
COPY . .
RUN chmod u+x scripts/entrypoint.sh

ENTRYPOINT sh ./scripts/entrypoint.sh -m $MODE
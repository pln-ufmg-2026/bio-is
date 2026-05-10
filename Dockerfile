FROM ubuntu:18.04

ENV WORKDIR=/e2sc-is
ENV PYTHONIOENCODING utf-8
ENV LANG="C.UTF-8"
ENV JOBLIB_TEMP_FOLDER=/tmp
ENV TMPDIR=/tmp
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo

WORKDIR $WORKDIR

# Install python 3.6 and OS dependencies
# Switch to US mirrors
RUN sed -i -e 's/archive.ubuntu.com/us.archive.ubuntu.com/g' -e 's/security.ubuntu.com/us.archive.ubuntu.com/g' /etc/apt/sources.list \
    && apt-get update -y --fix-missing \
    && apt-get install -y --no-install-recommends \
    build-essential \
    python3.6 python3.6-dev python3-pip \
    wget nano curl git ninja-build ccache libopenblas-dev libopencv-dev \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Linking python
RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && \
    ln -sfn /usr/bin/python3 /usr/bin/python && \
    ln -sfn /usr/bin/pip3 /usr/bin/pip

# Copy only requirements.txt first to leverage Docker cache for pip installs
COPY requirements.txt /e2sc-is/settings/

# Upgrading setuptools pip wheel and Installing requeriments
RUN python -m pip install --upgrade setuptools pip wheel \
    && python -m pip install -r /e2sc-is/settings/requirements.txt 

# Copy the rest of the files
COPY . /e2sc-is/settings/

#Install
#cd e2sc-is/settings/
#docker build -t e2sc:1.0 .
#Using
#cd ..
#docker run --rm --name e2sctest -v .:/e2sc-is -i -t e2sc:0.1 /bin/bash
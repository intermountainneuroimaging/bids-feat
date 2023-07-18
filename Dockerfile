# Creates docker container that runs HCP Pipeline algorithms
# Maintainer: Amy Hegarty (amy.hegarty@colorado.edu)
#

FROM ubuntu:focal as base
#
LABEL maintainer="Amy Hegarty <amy.hegarty@colorado.edu>"

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

ENV FSLDIR="/opt/fsl-6.0.4" \
    PATH="/opt/fsl-6.0.4/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLTCLSH="/opt/fsl-6.0.4/bin/fsltclsh" \
    FSLWISH="/opt/fsl-6.0.4/bin/fslwish" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"

RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bc \
           curl \
           ca-certificates\
           dc \
           file \
           libfontconfig1 \
           libfreetype6 \
           libgl1-mesa-dev \
           libgl1-mesa-dri \
           libglu1-mesa-dev \
           libgomp1 \
           libice6 \
           libxcursor1 \
           libxft2 \
           libxinerama1 \
           libxrandr2 \
           libxrender1 \
           libxt6 \
           sudo \
           wget \
           software-properties-common \
           dirmngr \
           ed \
           less \
           locales \
           vim-tiny \
           gpg-agent \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-6.0.4 \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.4-centos6_64.tar.gz \
    | tar -xz -C /opt/fsl-6.0.4 --strip-components 1


RUN echo "Installing FSL conda environment ..." \
    && python_install=/opt/fsl-6.0.4/etc/fslconf/fslpython_install.sh \
    && sed -i 's/dl_cmd_opts="--fail"/dl_cmd_opts="--fail -L"/g' $python_install \
    && bash $python_install -f /opt/fsl-6.0.4

######################################################
# FLYWHEEL GEAR STUFF...

USER root
RUN adduser --disabled-password --gecos "Flywheel User" flywheel

ENV USER="flywheel"

# Add poetry oversight.
RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
	 git \
     zip \
    software-properties-common &&\
	add-apt-repository -y 'ppa:deadsnakes/ppa' &&\
	apt-get update && \
	apt-get install -y --no-install-recommends python3.9\
    python3.9-dev \
	python3.9-venv \
	python3-pip &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install poetry based on their preferred method. pip install is finnicky.
# Designate the install location, so that you can find it in Docker.
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.1.6 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive questions
    POETRY_NO_INTERACTION=1 \
    VIRTUAL_ENV=/opt/venv
RUN python3.9 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3.9 -m pip install --upgrade pip && \
    ln -sf /usr/bin/python3.9 /opt/venv/bin/python3
ENV PATH="$POETRY_HOME/bin:$PATH"

# get-poetry respects ENV
RUN curl -sSL https://install.python-poetry.org | python3 - ;\
    ln -sf ${POETRY_HOME}/lib/poetry/_vendor/py3.9 ${POETRY_HOME}/lib/poetry/_vendor/py3.8; \
    chmod +x "$POETRY_HOME/bin/poetry"

# Installing main dependencies
ARG FLYWHEEL=/flywheel/v0
COPY pyproject.toml poetry.lock $FLYWHEEL/
WORKDIR $FLYWHEEL
RUN poetry install --no-root --no-dev

## Installing the current project (most likely to change, above layer can be cached)
## Note: poetry requires a README.md to install the current project
COPY run.py manifest.json README.md $FLYWHEEL/
COPY fw_gear_bids_feat $FLYWHEEL/fw_gear_bids_feat
COPY utils $FLYWHEEL/utils

# Configure entrypoint
RUN chmod a+x $FLYWHEEL/run.py && \
    echo "hcp-fsl-feat" > /etc/hostname && \
    rm -rf $HOME/.npm

ENTRYPOINT ["poetry","run","python","/flywheel/v0/run.py"]

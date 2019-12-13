#!/usr/bin/env bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh && \
    echo PATH=$HOME'/miniconda/bin:$PATH' >> ~/.bashrc && \
    $HOME/miniconda/bin/conda config --add channels conda-forge && \
    $HOME/miniconda/bin/conda update --yes -n base conda && \
    $HOME/miniconda/bin/conda update --all --yes

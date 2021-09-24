#!/usr/bin/bash
# USAGE: bash $0

set -ex
cd configs

cat << END > requirements.in
mecab-python3==0.996.5
tqdm
rank_bm25==0.2.1
torch==1.6.0
transformers==2.11.0
spacy==2.3.2
faiss-cpu==1.6.3
pandas==1.1.4
tensorboard==2.3.0

memory-profiler==0.58.0
matplotlib==3.3.3
END

pip install pip-tools
pip-compile requirements.in
pip-sync

cd ..

mkdir lib
cd lib
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

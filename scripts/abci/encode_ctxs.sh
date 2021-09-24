#!/bin/bash
# qsub -cwd -g gcb50246 -l rt_G.large=1 -l h_rt=12:00:00 -N log_enc -j y $HOME/exp2021/DPR_baseline/scripts/abci/encode_ctxs.sh

PATH=$HOME/.linuxbrew/bin:$HOME/.linuxbrew/sbin:$PATH

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"

pyenv activate miniconda3-3.19.0/envs/DPR_baseline

source /etc/profile.d/modules.sh
module load cuda/11.1/11.1.1 nccl/2.9/2.9.9-1 gcc/7.4.0 cudnn/8.2/8.2.0

export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

DATE=`date +%Y%m%d-%H%M`
echo $DATE

hostname
uname -a
which python
python --version
pip list


# =========================================

ROOT=$HOME/exp2021/DPR_baseline

exp_name=baseline
model=/groups/1/gcb50246/migrated_from_SFA_GPFS/miyawaki/DPR_baseline/baseline/retriever/dpr_biencoder.59.230.pt


bash $ROOT/scripts/retriever/encode_ctxs.sh \
  -n $exp_name \
  -m $model


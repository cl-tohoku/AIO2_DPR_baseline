#!/bin/bash
# qsub -cwd -g gcb50246 -l rt_G.large=1 -l h_rt=72:00:00 -N log_reader -j y $HOME/exp2021/DPR_baseline/scripts/abci/train_reader.sh

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
config_file=scripts/configs/reader_base.json
train_file=/groups/1/gcb50246/migrated_from_SFA_GPFS/miyawaki/DPR_baseline/baseline/retrieved/train_jaqket_59.230.json
dev_file=/groups/1/gcb50246/migrated_from_SFA_GPFS/miyawaki/DPR_baseline/baseline/retrieved/dev_jaqket_59.230.json


bash $ROOT/scripts/reader/train_reader.sh \
  -n $exp_name \
  -c $config_file \
  -t $train_file \
  -d $dev_file

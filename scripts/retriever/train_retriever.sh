#!/bin/bash
USAGE="bash $0 [-n NAME] [-c CONFIG] [-g GPU]"
DATE=`date +%Y%m%d-%H%M`

while getopts n:c:g: opt ; do
  case ${opt} in
    n ) FLG_N="TRUE"; NAME=${OPTARG};;
    c ) FLG_C="TRUE"; CONFIG=${OPTARG};;
    g ) FLG_G="TRUE"; GPU=${OPTARG};;
    * ) echo ${USAGE} 1>&2; exit 1 ;;
  esac
done

test "${FLG_N}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_C}" != "TRUE" && CONFIG=scripts/configs/retriever_base.json
test "${FLG_G}" == "TRUE" && export CUDA_VISIBLE_DEVICES=$GPU


# Train Retriever ======================================

set -ex
source scripts/configs/config.pth

DIR_PROJECT=$DIR_DPR/$NAME
mkdir -p $DIR_PROJECT/retriever
cp $CONFIG $DIR_PROJECT/retriever/hps.json
cp $0 $DIR_PROJECT/retriever/logs/run_${DATE}.sh

LOG_FILE=$DIR_PROJECT/retriever/logs/train_${DATE}.log
mkdir -p `dirname $LOG_FILE`
echo "# bash $0 $@" > $LOG_FILE

python train_dense_encoder.py \
  --train_file $TRAIN_FILE \
  --dev_file $DEV_FILE \
  --output_dir $DIR_PROJECT/retriever \
  --tensorboard_dir $DIR_PROJECT/retriever/tensorboard \
  --config $DIR_PROJECT/retriever/hps.json \
| tee $LOG_FILE

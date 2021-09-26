#!/bin/bash
USAGE="bash $0 [-n NAME] [-e EVAL_FILE] [-m MODEL] [-g GPU]"
DATE=`date +%Y%m%d-%H%M`

while getopts n:e:m:g: opt ; do
  case ${opt} in
    n ) FLG_N="TRUE"; NAME=${OPTARG};;
    e ) FLG_E="TRUE"; EVAL_FILE=${OPTARG};;
    m ) FLG_M="TRUE"; MODEL=${OPTARG};;
    g ) FLG_G="TRUE"; GPU=${OPTARG};;
    * ) echo ${USAGE} 1>&2; exit 1 ;;
  esac
done

test "${FLG_N}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_E}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_M}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_G}" == "TRUE" && export CUDA_VISIBLE_DEVICES=$GPU


# Eval Reader =============================

set -ex
source scipts/configs/config.pth

DIR_PROJECT=$DIR_DPR/$NAME
# mkdir -p $DIR_PROJECT/reader/results
# cp $CONFIG $DIR_PROJECT/reader/hps.json
# cp $0 $DIR_PROJECT/reader/run.sh

LOG_FILE=$DIR_PROJECT/logs/reader/eval_${DATE}.log
mkdir -p `dirname $LOG_FILE`
echo "# bash $0 $@" > $LOG_FILE

python train_reader.py \
    --dev_file $EVAL_FILE \
    --model_file $MODEL \
    --prediction_results_dir $DIR_PROJECT/reader/results \
    --config $DIR_PROJECT/reader/hps.json \
| tee -a $LOG_FILE

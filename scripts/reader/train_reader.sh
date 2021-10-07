#!/bin/bash
USAGE="bash $0 [-n NAME] [-c CONFIG] [-t TRAIN_READER_FILE] [-d DEV_READER_FILE] [-g GPU]"
DATE=`date +%Y%m%d-%H%M`

while getopts n:c:t:d:g: opt ; do
  case ${opt} in
    n ) FLG_N="TRUE"; NAME=${OPTARG};;
    c ) FLG_C="TRUE"; CONFIG=${OPTARG};;
    t ) FLG_T="TRUE"; TRAIN_FILE=${OPTARG};;
    d ) FLG_D="TRUE"; DEV_FILE=${OPTARG};;
    g ) FLG_G="TRUE"; GPU=${OPTARG};;
    * ) echo ${USAGE} 1>&2; exit 1 ;;
  esac
done

test "${FLG_N}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_C}" != "TRUE" && CONFIG=scripts/configs/reader_base.json
test "${FLG_T}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_D}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_G}" == "TRUE" && export CUDA_VISIBLE_DEVICES=$GPU


# Train Reader =============================

set -ex
source scripts/configs/config.pth

DIR_PROJECT=$DIR_DPR/$NAME
mkdir -p $DIR_PROJECT/reader -p $DIR_PROJECT/reader/tensorboard -p $DIR_PROJECT/reader/results
cp $CONFIG $DIR_PROJECT/reader/hps.json
cp $0 $DIR_PROJECT/reader/run.sh

LOG_FILE=$DIR_PROJECT/logs/reader/train_${DATE}.log
mkdir -p `dirname $LOG_FILE`
echo "# bash $0 $@" > $LOG_FILE

python train_reader.py \
  --train_file $TRAIN_READER_FILE \
  --dev_file $DEV_READER_FILE \
  --output_dir $DIR_PROJECT/reader \
  --tensorboard_dir $DIR_PROJECT/reader/tensorboard \
  --prediction_results_dir $DIR_PROJECT/reader/results \
  --config $DIR_PROJECT/reader/hps.json \
| tee -a $LOG_FILE

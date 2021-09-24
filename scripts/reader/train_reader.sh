#!/bin/bash
USAGE="bash $0 [-n NAME] [-c CONFIG] [-t TRAIN_FILE] [-d DEV_FILE] [-g GPU]"
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
DIR_MODEL=$DIR_PROJECT/outputs/reader/models
DIR_TENSORBOARD=$DIR_PROJECT/outputs/reader/tensorboard
DIR_RESULT=$DIR_PROJECT/outputs/reader/reults
DIR_LOG=$DIR_PROJECT/outputs/reader/logs
mkdir -p $DIR_PROJECT/outputs/reader $DIR_MODE $DIR_TENSORBOARD $DIR_LOG $DIR_RESULT
cp $CONFIG $DIR_PROJECT/outputs/reader/hps.json
cp $0 $DIR_PROJECT/outputs/reader/run.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_reader.py \
  --train_file $TRAIN_FILE \
  --dev_file $DEV_FILE \
  --output_dir $DIR_MODEL \
  --tensorboard_dir $DIR_TENSORBOARD \
  --prediction_results_dir $DIR_RESULT \
  --config $DIR_PROJECT/outputs/reader/hps.json \
| tee $DIR_LOG/train_${DATE}.log

#!/bin/bash
USAGE="bash $0 [-n NAME] [-e EVAL_FILE] [-g GPU] [-m MODEL_FILE]"
DATE=`date +%Y%m%d-%H%M`

while getopts n:c:e:g: opt ; do
  case ${opt} in
    n ) FLG_N="TRUE"; NAME=${OPTARG};;
    e ) FLG_E="TRUE"; EVAL_FILE=${OPTARG};;
    g ) FLG_G="TRUE"; GPU=${OPTARG};;
    m ) FLG_M="TRUE"; MODEL_FILE=${OPTARG};;
    * ) echo ${USAGE} 1>&2; exit 1 ;;
  esac
done

test "${FLG_N}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_E}" != "TRUE" && (echo ${USAGE} && exit 1)
test "${FLG_G}" == "TRUE" && export CUDA_VISIBLE_DEVICES=$GPU
test "${FLG_M}" != "TRUE" && (echo ${USAGE} && exit 1)


# Eval Reader =============================

set -ex
source scipts/configs/config.pth

DIR_PROJECT=$DIR_DPR/$NAME
DIR_RESULT=$DIR_PROJECT/outputs/reader/reults
DIR_LOG=$DIR_PROJECT/outputs/reader/logs
mkdir -p $DIR_PROJECT/outputs/reader $DIR_RESULT $DIR_LOG
cp $CONFIG $DIR_PROJECT/outputs/reader/hps.json
cp $0 $DIR_PROJECT/outputs/reader/run.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_reader.py \
    --dev_file $EVAL_FILE \
    --model_file $MODEL_FILE \
    --prediction_results_dir $DIR_RESULT \
    --config $DIR_PROJECT/reader/hps.json \
| tee $DIR_LOG/train_${DATE}.log
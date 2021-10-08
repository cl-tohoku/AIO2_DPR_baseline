#!/bin/bash
USAGE="bash $0 [-n NAME] [-m MODEL] [-g GPU]"
DATE=`date +%Y%m%d-%H%M`

while getopts n:m:p:g: opt ; do
  case ${opt} in
    n ) FLG_N="TRUE"; NAME=${OPTARG};;
    m ) FLG_M="TRUE"; MODEL=${OPTARG};;
    g ) FLG_G="TRUE"; GPU=${OPTARG};;
    * ) echo ${USAGE} 1>&2; exit 1 ;;
  esac
done

test "${FLG_N}" != "TRUE" && echo ${USAGE} 1>&2 && exit 1
test "${FLG_M}" != "TRUE" && echo ${USAGE} 1>&2 && exit 1
test "${FLG_G}" == "TRUE" && export CUDA_VISIBLE_DEVICES=$GPU


# Encode Passages ======================================

set -ex

source scripts/configs/config.pth
DIR_PROJECT=$DIR_DPR/$NAME

LOG_FILE=$DIR_PROJECT/embeddings/logs/embs_${DATE}.log
mkdir -p `dirname $LOG_FILE`
echo "# bash $0 $@" > $LOG_FILE

python generate_dense_embeddings.py \
  --batch_size 512 \
  --model_file $MODEL \
  --ctx_file $WIKI_FILE \
  --output_dir $DIR_PROJECT/embeddings \
| tee -a $LOG_FILE

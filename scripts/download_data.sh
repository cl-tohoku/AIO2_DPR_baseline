#!/bin/bash
USAGE="bash $0 [output_dir]"
DATE=`date +%Y%m%d-%H%M`

set -e

DEST=$1

if [ -z $DEST ] ; then
  echo "[ERROR] Please specify the 'output_dir'"
  echo $USAGE
  exit 1
fi


# DPR
mkdir -p $DEST/aio
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/retriever/abc_01-12.json.gz -P $DEST/aio
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/retriever/aio_01_dev.json.gz -P $DEST/aio
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/retriever/aio_01_test.json.gz -P $DEST/aio
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/retriever/aio_01_unused.json.gz -P $DEST/aio

wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/qas/abc_01-12.tsv -P $DEST/aio
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/qas/aio_01_dev.tsv -P $DEST/aio
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/qas/aio_01_test.tsv -P $DEST/aio
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/qas/aio_01_unused.tsv -P $DEST/aio


# wikipedia
mkdir -p $DEST/wiki
wget -nc https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/wikipedia_split/jawiki-20210503-paragraphs.tsv.gz -P $DEST/wiki



cat << END > scripts/configs/config.pth
# data
WIKI_FILE=$DEST/wiki/jawiki-20210503-paragraphs.tsv.gz
TRAIN_FILE=$DEST/aio/abc_01-12.json.gz
DEV_FILE=$DEST/aio/aio_01_dev.json.gz
TEST_FILE=$DEST/aio/aio_01_test.json.gz

# dest (To create models, embeddings, etc under `$DIR_DPR/$NAME`)
DIR_DPR=outputs
END


echo -en "\n===========================================\n"
ls -R -lh $DEST

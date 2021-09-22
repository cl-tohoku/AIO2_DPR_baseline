#!/usr/bin/bash
USAGE="bash $0 [DEST]"

set -ex

if [ -z $1 ] ; then
    echo "Please specify the destination dir."
    echo $USAGE
    exit 1
fi


DEST=$1
mkdir -p $DEST/wiki $DEST/aio

wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/train_questions.json -P $DEST/aio
wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev1_questions.json -P $DEST/aio
wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev2_questions.json -P $DEST/aio
wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/candidate_entities.json.gz -P $DEST/wiki


cat << END > scripts/configs/config.pth
# dest (To create models, embeddings, etc under `$DIR_DPR/$NAME`)
DIR_DPR=outputs

# path
WIKI_FILE=$DEST/wiki/jawiki-20210503-paragraphs.tsv.gz
TRAIN_FILE=$DEST/aio/abc_eqiden_01-12.json.gz
DEV_FILE=$DEST/aio/aio_2020_dev.json.gz
TEST_FILE=$DEST/aio/aio_2020_test.json.gz
END

ls -R $DEST

echo "DONE"
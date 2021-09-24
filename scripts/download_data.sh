#!/usr/bin/bash
set -e
USAGE="USAGE: bash $0 [jaqket/rcqa] [DEST]"
. configs/config.sh


if [ $1 = "jaqket" ] ; then

    [ ! -z $2 ] && DEST=$2 || DEST=$DIR_DATA/source/jaqket
    mkdir -p $DEST

    wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/train_questions.json -P $DEST
    wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev1_questions.json -P $DEST
    wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/dev2_questions.json -P $DEST
    wget -nc https://jaqket.s3-ap-northeast-1.amazonaws.com/data/candidate_entities.json.gz -P $DEST
    wget -nc https://www.nlp.ecei.tohoku.ac.jp/projects/AIP-LB/static/aio_leaderboard.json -P $DEST

    mv $DEST/aio_leaderboard.json $DEST/test_questions.json
    echo "https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/" > $DEST/README.md

    echo -e "${GREEN}`ls $DEST`${END}"


elif [ $1 = "rcqa" ] ; then

    [ ! -z $2 ] && DEST=$2 || DEST=$DIR_DATA/source/rcqa
    mkdir -p $DEST

    wget -nc http://www.cl.ecei.tohoku.ac.jp/rcqa/data/all-v1.0.json.gz -P $DEST
    
    echo "http://www.cl.ecei.tohoku.ac.jp/rcqa/" > $DEST/README.md
    echo -e "${GREEN}`ls $DEST`${END}"

else
    echo $USAGE
    exit 1

fi



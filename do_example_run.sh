#!/bin/bash
<< COMMENT
本スクリプトは README の実行手順に倣い、
データダウンロードから reader による推論までの一連の流れを記したものです。
README で記載している retriever の Acc@k や reader の Acc については以下の実行手順で出力されたものになります。
オプションで指定するモデルやデータについては各自適切なものを指定して下さい。
============================================================================

- python = 3.8.11
- cuda/11.1/11.1.1
- nccl/2.9/2.9.9-1
- gcc/7.4.0
- cudnn/8.2/8.2.0

COMMENT


### ダウンロード ###
bash scripts/download_data.sh datasets


### 1. BiEncoder の学習 ###
bash scripts/retriever/train_retriever.sh \
  -n exp1 \
  -c scripts/configs/retriever_base.json


### 2. 文書集合のエンコード ###
bash scripts/retriever/encode_ctxs.sh \
  -n exp1 \
  -m outputs/exp1/retriever/dpr_biencoder.59.230.pt


### 3. データセットの質問に関連する文書抽出 ###
bash scripts/retriever/retrieve_passage.sh \
  -n exp1 \
  -m outputs/exp1/retriever/dpr_biencoder.59.230.pt \
  -e outputs/exp1/embeddings/emb_dpr_biencoder.59.230.pickle

# retriever による正解率
cat outputs/exp1/retrieved/train_jaqket_59.230.tsv
cat outputs/exp1/retrieved/dev_jaqket_59.230.tsv
cat outputs/exp1/retrieved/test_jaqket_59.230.tsv


### 4. Reader の学習 ###
bash scripts/reader/train_reader.sh \
  -n exp1 \
  -c scripts/configs/reader_base.json \
  -t outputs/exp1/retrieved/train_jaqket_59.230.json \
  -d outputs/exp1/retrieved/dev_jaqket_59.230.json


### 5. 評価 ###
bash scripts/reader/eval_reader.sh \
  -n exp1 \
  -e outputs/exp1/retrieved/dev_jaqket_59.230.json \
  -m outputs/exp1/reader/dpr_reader_best.pt

bash scripts/reader/eval_reader.sh \
  -n exp1 \
  -e outputs/exp1/retrieved/test_jaqket_59.230.json \
  -m outputs/exp1/reader/dpr_reader_best.pt

# reader による正解率
cat outputs/exp1/reader/results/eval_accuracy.txt


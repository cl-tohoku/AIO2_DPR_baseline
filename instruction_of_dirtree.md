# 各ファイルの概要

```yaml
# `AIO2_DPR_baseline/` 下に存在する DPR を学習するための python コード
- train_dense_encoder.py:       (1) retriever を学習するコード
- generate_dense_embedding.py:  (2) 学習済みの retriever を使用して Wikipedia 文書から embeddings を作成するコード
- dense_retriever.py:           (3) 学習済みの retriever および作成した embeddings を使用して reader の学習ファイルを作成するコード
- train_reader.py:              (4) reader を学習・評価するコード

# DPR に関する python コード
- dpr/:
  - data/:
    - qa_validation.py:         retriever および reader を評価する際に使用する関数がまとめられているコード
    - reader_data.py:           reader 学習用にデータを前処理する際に使用する関数がまとめられているコード
  - indexer/:
    - faiss_indexers.py:        retriever を用いたオフライン検索時に使用する faiss に関するコード
  - models/:
    - __init__.py:              コマンドライン引数 `--encoder_model_type` から指定されたモデルを選択するためのコード
    - biencoder.py:             retriever に使用される biencoer クラスとロス関数（`BiEncoderNllLoss`）が定義される
    - hf_models.py:             huggingface をベースとしたエンコーダクラスの wrapper が定義される（`--encoder_model_type=hf_bert`）
    - fairseq_models.py:        fairseq をベースとしたエンコーダクラスの wrapper が定義される（`--encoder_model_type=fairseq_roberta`）
    - pytext_models.py:         pytext をベースとしたエンコーダクラスの wrapper が定義される（`--encoder_model_type=pytext_bert`）
    - reader.py:                reader のクラスやロス関数などが定義される
  - utils/:
    - data_utils.py:            データ用 utils
    - dist_utils.py:            分散学習用 utils
    - model_utils.py:           モデル用 utils
    - tokenizers.py:            トークナイザがまとめられたコード（BertJapaneseTokenizer は `dpr/models/hf_models.py` で定義される）
  - options.py:                 コマンドライン引数（argparse）で指定する引数がまとめられているコード

# コマンド実行用シェルスクリプト
- scripts/:
  - configs/:
    - config.pth:               データセットのディレクトリやモデルの出力先ディレクトリを記載する
    - retriever_base.json:      ベースラインにおける retriever のパラメータが記載される
    - reader_base.json:         ベースラインにおける reader のパラメータが記載されているファイル
  - retriever/:
    - train_retriever.sh:       (1) retriever の学習を実行する
    - encode.ctxs.sh:           (2) 学習済みの retriever を使用して Wikipedia 文書に対する embeddings の作成を実行する
    - retrieve_passgae.sh:      (3) 学習済みの retriever および作成した embeddings を使用して reader の学習ファイルの作成を実行する
  - reader/:
    - train_reader.sh:          (4) reader の学習を実行する
    - eval_reader.sh:           (4) reader の評価を実行する
  - download_data.sh:           配布用のデータセットのダウンロードを実行する
  - dowmload_model.sh:          配布用の学習済みモデルのダウンロードを実行する

# その他のディレクトリ
- data/:                        データセット作成時に使用したスクリプトが含まれるディレクトリ
- model/:                       docker image 提出時に実行する `docker run` のマウント対象となるディレクトリ
```

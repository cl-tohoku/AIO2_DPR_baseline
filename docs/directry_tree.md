# 各ディレクトリおよびスクリプトの概要について


```yaml
# 一部省略

- docs/: 各種ドキュメント
- data/: データセット作成時に使用した実行スクリプト
- model/: `docker run`実行時のマウント対象となるディレクトリ

# 実行スクリプト（bash）
- scripts/:
  - configs/:
    - config.pth: データや保存先に関するパスの設定ファイル
    - retriever_base.json: retriever のベースライン設定
    - reader_base.json: reader のベースライン設定
  - retriever/:
    - train_retriever.sh: retriever 学習用スクリプト
    - encode_ctxs.sh: retriever による wikipedia ファイルのエンコード用スクリプト
    - retrieve_passage.sh: QA データに対する関連文書検索用スクリプト
  - reader/:
    - train_reader.sh: reader 学習用スクリプト
    - eval_reader.sh: reader 評価用スクリプト
  - download_data.sh: データセットダウンロード用スクリプト
  - download_model.sh: 学習済みモデルダウンロード用スクリプト

# dpr に関するスクリプト
- dpr/:
  - data/:
    - qa_validation.py: 
    - reader_data.py: 
  - indexer/:
    - faiss_indexers.py: retriever による検索時に使用する faiss に関するスクリプト
  - models/:
    - __init__.py: モデル選択時に使用されるスクリプト
    - biencoder.py: retriever に使用される biencoder 用モデルスクリプト
    - hf_models.py: huggingface BERT 用モデルスクリプト
    - reader.py: reader モデルスクリプト
  - utils/:
    - data_utils.py: データ用 utils
    - dist_utils.py: 分散学習用 utils
    - model_utils.py: モデル用 utils
  - options.py: argparse で指定するオプションに関するスクリプト

```

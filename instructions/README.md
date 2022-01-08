## 各ファイルの説明は以下の通りです．

```bash
|- train_dense_encoder.py   # retriever を学習するコード
|- generate_dense_embedding.py   # 学習済みのretriever model を使用してWikipedia 文書をembedding 化するコード
|- dense_retroever.py   # 訓練済みのretriever model と作成したembedding を用いてreader 学習・評価用のファイルを作成するコード
|- train_readet.py  # reader を学習・評価するコード
|- download_data.sh # データセットをダウンロードするshell script
|- dowmload_model.sh    # 学習済みモデルをダウンロードするshell script
|- data # 詳細は data ディレクトリ内のREADME.md に記載されています
|- dpr/
|- |- data/
|- |- |- qa_validation.py   # reeder の結果を評価する際に使用する関数がまとめられているコード
|- |- |- reader_data.py # reader 用にデータを前処理する際に使用する関数がまとめられているコード
|- |- indexer/
|- |- |- faiss_indexers.py  # FAISS を用いてWikipedia文書をindex 化するコード
|- |- modles/
|- |- |- __init__.py    # モジュールのimportを初期化するコード
|- |- |- biencoder.py   # Biencoer モジュールとロス関数がまとめられたコード
|- |- |- fairseq_models.py   # Fairseq をベースとしたencoder モジュールのwrapper がまとめられているコード
|- |- |- hf_models.py   # Huggingface をベースとしたencoder モジュールのwrapper がまとめられているコード
|- |- |- pytext_models.py   # Pytext をベースとしたencoder モジュールのwrapper がまとめられているコード
|- |- |- reader.py  # Reader class やロス関数などがまとめられているコード
|- |- utils/
|- |- |- options.py # コマンドライン上で指定する引数がまとめられているコード
|- scripts/
|- |- configs
|- |- |- config.path    # データセットのディレクトリやモデルの出力ディレクトリが記載されているファイル
|- |- |- reader_base.json   # reader のパラメータが記載されているファイル
|- |- |- retriever_base.json    # retriever のパラメータが記載されているファイル
|- |- reader
|- |- |- train_reader.sh    # reder を学習するshell script
|- |- |- eval_reader.sh # reder を評価するshell script
|- |- retriever
|- |- |- train_retriever.sh # retriever を学習するshell script
|- |- |- encode.ctxs.sh #  Wikipedia 文書のembedding を作成するshell script
|- |- |- retrieve_passgae.sh    # reader 学習・評価用のデータセットを作成するshell script
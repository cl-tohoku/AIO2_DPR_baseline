## モデルの動作に必要なデータセットについて

本リポジトリに含まれる DPR のモデルの動作には、オリジナルの DPR と同様のフォーマットの各種データセットが必要です。
以下では、「AI王」のクイズ問題と Wikipedia の記事段落を利用した、 DPR 形式のデータセットの作成方法について説明します。

**注意:** これらのデータセットは `/scripts/download_data.sh` を実行することで、作成済みのものがダウンロードされます。モデルの動作のために以下の手順をあらためて実行する必要はありません。

### 必要なもの

- Wikipedia Cirrussearch ダンプファイル
    - 本データセットの作成には2021年5月13日付のファイルである `jawiki-20210503-cirrussearch-content.json.gz` を使用しました。
    - 最新のものは https://dumps.wikimedia.org/other/cirrussearch/ より入手できます。
- [Elasticsearch](https://www.elastic.co/jp/elasticsearch/) と以下のプラグイン
    - [ICU Analysis plugin](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu.html)
    - [Japanese (kuromoji) Analysis plugin](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-kuromoji.html)
    - 本データセットの作成には Elasticsearch v6.5.4 を使用しました。

### データセット作成手順

1. Wikipedia Cirrussearch ダンプファイルから、Wikipedia 記事のページ ID を取得します。

```sh
$ python get_all_page_ids_from_cirrussearch.py \
--cirrus_file <data_dir>/jawiki-20210503-cirrussearch-content.json.gz \
--output_file <work_dir>/jawiki-20210503-pageids.json \
--min_inlinks 10 \
--exclude_sexual_pages
```

2. [Wikipedia REST API](https://en.wikipedia.org/api/rest_v1/) を用いて、 Wikipedia 記事の HTML を取得します。

**注意:** すべての Wikipedia 記事の HTML の取得には、およそ2日以上の時間を要します。

```sh
$ python get_page_htmls.py \
--page_ids_file <work_dir>/jawiki-20210503-pageids.json \
--output_file <work_dir>/jawiki-20210503-page-htmls.json.gz \
--batch_size 10
```

3. Wikipedia 記事の HTML から、本文の段落を抽出します。

```sh
$ python extract_paragraphs_from_page_htmls.py \
--input_file <work_dir>/jawiki-20210503-page-htmls.json.gz \
--output_file <work_dir>/jawiki-20210503-paragraphs.json.gz \
--min_text_length 10 \
--max_text_length 1000
```

4. 抽出された記事段落から、長さが一定文字数以内であるものを取り出し、 DPR で扱うパッセージとします。

```sh
$ python make_passages_from_paragraphs.py \
--input_file <work_dir>/jawiki-20210503-paragraphs.json.gz \
--output_file <work_dir>/passages-jawiki-20210503-paragraphs.json.gz \
--max_passage_length 400
```

5. 作成されたパッセージを用いて、 Elasticsearch のインデックスを作成します。

```sh
$ python build_es_index_passages.py \
--input_file <work_dir>/passages-jawiki-20210503-paragraphs.json.gz \
--index_name jawiki-20210503-paragraphs \
--hostname localhost \
--port 9200
```

6. 質問応答データの質問に対して、適合度が高いパッセージを検索して付与します。

```sh
$ python make_dpr_retriever_dataset.py \
--input_file <data_dir>/abc_01-12.jsonl \
--output_file <dpr_retriever_data_dir>/abc_01-12.jsonl.gz \
--es_index_name jawiki-20210503-paragraphs \
--num_documents_per_question 100

$ python make_dpr_retriever_dataset.py \
--input_file <data_dir>/aio_01_dev.jsonl \
--output_file <dpr_retriever_data_dir>/aio_01_dev.jsonl.gz \
--es_index_name jawiki-20210503-paragraphs \
--num_documents_per_question 100

$ python make_dpr_retriever_dataset.py \
--input_file <data_dir>/aio_01_test.jsonl \
--output_file <dpr_retriever_data_dir>/aio_01_test.jsonl.gz \
--es_index_name jawiki-20210503-paragraphs \
--num_documents_per_question 100
```

7. 質問と正解のペアのデータを DPR で使われているフォーマットに変換します。

```sh
$ python make_dpr_qas_dataset.py \
--input_file <data_dir>/abc_01-12.jsonl \
--output_file <dpr_qas_data_dir>/abc_01-12.tsv

$ python make_dpr_qas_dataset.py \
--input_file <data_dir>/aio_01_dev.jsonl \
--output_file <dpr_qas_data_dir>/aio_01_dev.tsv

$ python make_dpr_qas_dataset.py \
--input_file <data_dir>/aio_01_test.jsonl \
--output_file <dpr_qas_data_dir>/aio_01_test.tsv
```

8. パッセージのデータを DPR で使われているフォーマットに変換します。

```sh
$ python make_dpr_wikipedia_split_dataset.py \
--input_file <work_dir>/passages-jawiki-20210503-paragraphs.json.gz \
--output_file <dpr_wikipedia_split_data_dir>/jawiki-20210503-paragraphs.tsv.gz
```

### ライセンス

本ディレクトリ内のすべてのスクリプトのライセンスは Apache License 2.0 とします。

このディレクトリは、推論システムの Docker イメージが参照するディレクトリです。
以下のファイルを規定のファイル名で配置することにより、推論システムの Docker イメージが動作します。

- `biencoder.pt`:
    - biencoder の訓練済みファイル
- `reader.pt`:
    - reader の訓練済みファイル
- `embedding.pickle`:
    - 文書エンベッディングファイル
- `passages.tsv.gz`:
    - 文書集合

例えば、 [../do_example_run.sh](../do_example_run.sh) の実行結果がある場合は、以下の操作により必要なファイルがコピーされます。

```sh
$ cp ../outputs/exp1/retriever/dpr_biencoder.59.230.pt biencoder.pt
$ cp ../outputs/exp1/reader/dpr_reader_best.pt reader.pt
$ cp ../outputs/exp1/embeddings/emb_dpr_biencoder.59.230.pickle embedding.pickle
$ cp ../datasets/wiki/jawiki-20210503-paragraphs.tsv.gz passages.tsv.gz
```

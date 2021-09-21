![AIO](imgs/aio.png)

# AI王 〜クイズAI日本一決定戦〜 2021
オープンドメイン質問応答
- [AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/)
- 昨年度の概要は [こちら](#)

## 目次

- [Setup](#Setup)
    - 環境構築
    - データセット


## セットアップ

```bash
$ pip install pip-tools
$ pip-compile requirements.in
$ pip-sync
```

## データセット

> __JAQKET: クイズを題材にした日本語QAデータセット__
> - https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/
> - 鈴木正敏, 鈴木潤, 松田耕史, ⻄田京介, 井之上直也. JAQKET:クイズを題材にした日本語QAデータセットの構築. 言語処理学会第26回年次大会(NLP2020) [\[PDF\]](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf)

### Download

```bash
$ bash scripts/download_data.sh <output_dir>

<output_dir>
|- wiki/
|  |- jawiki-20210503-paragraphs.tsv
|- aio/
|  |- abc_eqiden_01-12.json
|  |- dev_jaqket.json
|  |- test_jaqket.json
```

#### Statistics

|データ|質問数|
|:---|---:|
|訓練||
|開発||
|評価||
|wiki|6795533|

#### Jaqket Format

The expected data format is a list of entry examples, where each entry example is a dictionary containing

- `question`: question text
- `answers`: list of answer text
- `positive_ctxs`: a list of positive passages where each item is a dictionary containing following:
    - `id`: passage id
    - `title`: article title
    - `text`: passage text
- `negative_ctxs`: a list of negative passages (train 内で定義される)
- `hard_negative_ctxs`: a list of hard negative passages

```json
{
    "question": "明治時代に西洋から伝わった「テーブル・ターニング」に起源を持つ占いの一種で、50音表などを記入した紙を置き、参加者全員の人差し指をコインに置いて行うのは何でしょう?",
    "answers": [
        "コックリさん"
    ],
    "positive_ctxs": [
        {
            "id": 278397,
            "title": "コックリさん",
            "text": "コックリさん(狐狗狸さん)とは、西洋の「テーブル・ターニング(Table-turning)」に起源を持つ占いの一種。机に乗せた人の手がひとりでに動く現象は心霊現象だと古くから信じられていた。科学的には意識に関係なく体が動くオートマティスムの一種と見られている。「コックリさん」と呼ばれるようになったものは、日本で19世紀末から流行したものだが、これは「ウィジャボード」という名前の製品が発売されたりした海外での流行と同時期で、外国船員を通して伝わったという話がある。"
        }
    ],
    "negative_ctxs": [],
    "hard_negative_ctxs": [
        {
            "id": 3943003,
            "title": "星座占い",
            "text": "喫茶店などのテーブル上には、星座占いの機械が置かれていることがある。硬貨を投入して、レバーを動かすと、占いの内容が印刷された用紙が排出される。"
        },
    ]
}
```

#### Wikipedia Data Format

The wikipedia data is a format of tsv, where each entry example is a dictionary containing. 

```tsv
id      text    title
1       "モルガナイト(morganite)はピンク色ないし淡赤紫色の緑柱石(ベリル)である。呈色はマンガン(Mn)に由来する。" モルガナイト
```


## Dense Passage Retrieval 

![](imgs/dpr.png)


### Settings

```bash
$ vim scripts/configs/config.pth
```

### Retriever

#### 1. 学習

```bash
# bash scripts/retriever/train_retriever.sh

$ python train_dense_encoder.py \
  --train_file $TRAIN_FILE \
  --dev_file $DEV_FILE \
  --output_dir $DIR_PROJECT/retriever \
  --config $DIR_PROJECT/retriever/hps.json
```

#### 2. 文書集合のエンコード

```bash
# bash scripts/retriever/encode_ctxs.sh
$ python generate_dense_embeddings.py \
  --batch_size 512 \
  --model_file <model> \
  --ctx_file <wikipedia> \
  --output_dir $DIR_PROJECT/embeddings
```

#### 3. データセットの質問に関連する文書抽出

```bash
# bash scripts/retriever/retrieve_passage.sh

$ python dense_retriever.py \
    --n-docs 100 \
    --validation_workers 32 \
    --batch_size 64 \
    --projection_dim 768 \
    --model_file $MODEL \
    --ctx_file $WIKI_FILE \
    --encoded_ctx_file $EMBEDDING \
    --qa_file ${QA_FILES[$KEY]} \
    --out_file $FO_FILE
```

### Reader

#### 4. 学習

```bash
# bash sscripts/raeder/train_reader.sh

$ python ${WORK_DIR}/src/reader_train.py \
    --train_file ${FI_TRAIN} \
    --dev_file ${FI_DEV} \
    --output_dir ${MODEL_DIR} \
    --dir_tensorboard ${TENSORBOARD_DIR} \
    --loss_and_score_results_dir ${OUT_DIR} \
    --prediction_results_dir ${PREDICTION_RESULTS_DIR} \
```

#### 5. 評価

```bash
# bash sscripts/raeder/eval_reader.sh

$ python ${WORK_DIR}/src/reader_train.py \
    --dev_file ${FI_TEST} \
    --output_dir ${MODEL_DIR} \
    --dir_tensorboard ${TENSORBOARD_DIR} \
    --loss_and_score_results_dir ${OUT_DIR} \
    --prediction_results_dir ${PREDICTION_RESULTS_DIR} \
```

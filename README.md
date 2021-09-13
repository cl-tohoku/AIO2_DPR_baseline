# Dense Passage Retrieval

* [acl2020 open-QA Tutorial](https://github.com/danqi/acl2020-openqa-tutorial/blob/master/slides/part5-dense-retriever-e2e-training.pdf)


Dense Passage Retrieval (`DPR`) - is a set of tools and models for state-of-the-art open-domain Q&A research.
It is based on the following paper:


Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), Preprint 2020.

If you find this paper or this code useful, please cite this paper:
```
@article{karpukhin2020dense,
  title={Dense Passage Retrieval for Open-Domain Question Answering},
  author={Karpukhin, Vladimir and O{\u{g}}uz, Barlas and Min, Sewon and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau},
  journal={arXiv preprint arXiv:2004.04906},
  year={2020}
}
```

## Features
1. Dense retriever model is based on bi-encoder architecture.
2. Extractive Q&A reader&ranker joint model inspired by [this](https://arxiv.org/abs/1911.03868) paper.
3. Related data pre- and post- processing tools.
4. Dense retriever component for inference time logic is based on FAISS index.


## Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
git clone git@github.com:facebookresearch/DPR.git
cd DPR
pip install .
```

DPR is tested on Python 3.6+ and PyTorch 1.2.0+.
DPR relies on third-party libraries for encoder code implementations.
It currently supports Huggingface BERT, Pytext BERT and Fairseq RoBERTa encoder models.
Due to generality of the tokenization process, DPR uses Huggingface tokenizers as of now. So Huggingface is the only required dependency, Pytext & Fairseq are optional.
Install them separately if you want to use those encoders.



## Resources & Data formats
First, you need to prepare data for either retriever or reader training.
Each of the DPR components has its own input/output data formats. You can see format descriptions below.
DPR provides NQ & Trivia preprocessed datasets (and model checkpoints) to be downloaded from the cloud using our data/download_data.py tool. One needs to specify the resource name to be downloaded. Run 'python data/download_data.py' to see all options.

```bash
python data/download_data.py \
	--resource {key from download_data.py's RESOURCES_MAP}  \
	[optional --output_dir {your location}]
```
The resource name matching is prefix-based. So if you need to download all data resources, just use --resource data.

## Retriever input data format
The data format of the Retriever training data is JSON.
It contains pools of 2 types of negative passages per question, as well as positive passages and some additional information.

```
# datasets/data/retriever/nq-dev.json

[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"text": "...."
	}],
	"negative_ctxs": ["..."],
	"hard_negative_ctxs": ["..."]
  },
  ...
]
```

Elements' structure  for negative_ctxs & hard_negative_ctxs is exactly the same as for positive_ctxs.
The preprocessed data available for downloading also contains some extra attributes which may be useful for model modifications (like bm25 scores per passage). Still, they are not currently in use by DPR.
> negative_ctxs と hard_negative_ctxs の要素構造は positive_ctxs と全く同じです．ダウンロード可能な前処理データには，モデルを修正するのに有用ないくつかの追加属性が含まれています（一節あたりのBM25スコアのような）．しかし，これらの属性は現在のところDPRでは使用されていません．

You can download prepared NQ dataset used in the paper by using 'data.retriever.nq' key prefix. Only dev & train subsets are available in this format.
We also provide question & answers only CSV data files for all train/dev/test splits. Those are used for the model evaluation since our NQ preprocessing step looses a part of original samples set.
Use 'data.retriever.qas.\*' resource keys to get respective sets for evaluation.
> data.retriever.nq'という接頭辞をつけることで、論文で使用したNQデータセットをダウンロードすることができます。このフォーマットでは、devとtrainのサブセットのみが利用可能です。また、train/dev/testのすべての分割について、質問と回答のみのCSVデータファイルも提供しています。NQの前処理では、元のサンプルセットの一部が失われるため、これらはモデルの評価に使用されます。評価のためにそれぞれのセットを取得するには、'data.retriever.qas.*'リソースキーを使用してください。

```bash
python data/download_data.py
	--resource data.retriever
	[optional --output_dir {your location}]
```


## Retriever training
Retriever training quality depends on its effective batch size. The one reported in the paper used 8 x 32GB GPUs.
In order to start training on one machine:
```bash
python train_dense_encoder.py
	--encoder_model_type {hf_bert | pytext_bert | fairseq_roberta}
	--pretrained_model_cfg {bert-base-uncased| roberta-base}
	--train_file {train files glob expression}
	--dev_file {dev files glob expression}
	--output_dir {dir to save checkpoints}
```

Notes:
- If you use pytext_bert or fairseq_roberta, you need to download pre-trained weights and specify --pretrained_file parameter. Specify the dir location of the downloaded files for 'pretrained.fairseq.roberta-base' resource prefix for RoBERTa model or the file path for pytext BERT (resource name 'pretrained.pytext.bert-base.model').
- `--eval_per_epoch`: saving checkpoints per epoch
- train に関して epoch 以外に stop condition はない
- Every evaluation saves a model checkpoint.
- The best checkpoint is logged in the train process output.
- Regular NLL classification loss validation for bi-encoder training can be replaced with average rank evaluation. It aggregates passage and question vectors from the input data passages pools, does large similarity matrix calculation for those representations and then averages the rank of the gold passage for each question. We found this metric more correlating with the final retrieval performance vs nll classification loss. Note however that this average rank validation works differently in DistributedDataParallel vs DataParallel PyTorch modes. See `val_av_rank_` set of parameters to enable this mode and modify its settings.
> Bi-encoder 学習のための通常の NLL分類損失検証 を平均ランク評価に置き換えることができます。これは、入力データの passage pool から passage vec と question vec を集約し、それらの表現に対して大規模な類似度行列計算を行い、各 question に対する gold passage のランクを平均化します。我々は、この指標が最終的な検索性能とnll分類損失との相関性が高いことを発見した。ただし、この平均ランクの検証は、DistributedDataParallelとDataParallel PyTorchのモードでは動作が異なることに注意してください。このモードを有効にしたり、設定を変更したりするには、val_av_rank\_*パラメータセットを参照してください。

See the section 'Best hyperparameter settings' below as e2e example for our best setups.

## Retriever inference

Generating representation vectors for the static documents dataset is a highly parallelizable process which can take up to a few days if computed on a single GPU. You might want to use multiple available GPU servers by running the script on each of them independently and specifying their own shards.
> 静的文書データセットの表現ベクトルの生成は高度に並列化可能なプロセスであり、単一のGPUで計算した場合、最大で数日かかる可能性があります。複数の利用可能なGPUサーバを使用するには、それぞれのサーバでスクリプトを独立して実行し、それぞれのサーバのシャードを指定するとよいでしょう。

```bash
python generate_dense_embeddings.py \
	--model_file {path to biencoder checkpoint} \
	--ctx_file {path to psgs_w100.tsv file} \
	--shard_id {shard_num, 0-based} --num_shards {total number of shards} \
	--out_file ${out files location + name PREFX}
```
Note: you can use much large batch size here compared to training mode. For example, setting --batch_size 128 for 2 GPU(16gb) server should work fine.
You can download already generated wikipedia embeddings (trained on NQ dataset) using resource key 'data.retriever_results.nq.single.wikipedia_passages'.
> 注意: ここではトレーニングモードに比べて大きなバッチサイズを使用することができます。例えば、--batch_size 128で2GPU(16gb)のサーバであれば問題なく動作するはずです。リソースキー'data.retriever_results.nq.single.wikipedia_passages'を使って、既に生成されたwikipediaの埋め込みをダウンロードすることができる。

## Retriever validation against the entire set of documents:

```bash
python dense_retriever.py \
	--model_file ${path to biencoder checkpoint} \
	--ctx_file  {path to all documents .tsv file} \
	--qa_file {path to test|dev .csv file} \
	--encoded_ctx_file "{encoded document files glob expression}" \
	--out_file {path to output json file with results} \
  --n-docs 200
```

The tool writes retrieved results for subsequent reader model training into specified out_file.
It is a json with the following format:

```
[
    {
        "question": "...",
        "answers": ["...", "...", ... ],
        "ctxs": [
            {
                "id": "...", # passage id from database tsv file
                "title": "",
                "text": "....",
                "score": "...",  # retriever score
                "has_answer": true|false
     },
]
```
Results are sorted by their __similarity score__, from most relevant to least relevant.

By default, dense_retriever uses exhaustive search process, but you can opt in to use HNSW FAISS index by --hnsw_index flag.
Note that using this index may be useless from the research point of view since their fast retrieval process comes at the cost of much longer indexing time and higher RAM usage.
The similarity score provided is the dot product in the (default) case of exhaustive search and L2 distance in a modified representations space in case of HNSW index.
> デフォルトでは、dense_retrieverは網羅的な検索処理を行いますが、`--hnsw_index` フラグを設定することで、HNSW FAISSインデックスを使用することができます。このインデックスは、高速な検索処理の代償として、インデックス作成時間が長くなり、RAMの使用量が多くなるため、研究の観点からは役に立たないかもしれないことに注意してください。提供される類似度スコアは、(デフォルトの)網羅的検索の場合はドット積であり、HNSWインデックスの場合は修正された表現空間におけるL2距離である。


## Optional reader model input data pre-processing.
Since the reader model uses a specific combination of positive and negative passages for each question and also needs to know the answer span location in the bpe-tokenized form, it is recommended to preprocess and serialize the output from the retriever model before starting the reader training. This saves hours at train time.
If you don't run this preprocessing, the Reader training pipeline checks if the input file(s) extension is .pkl and if not, preprocesses and caches results automatically in the same folder as the original files.
> reader model は各質問に対して正と負のパッセージの特定の組み合わせを使用し、またbpe-tokenized形式の回答スパンの位置を知る必要があるため、reader model を開始する前に retrieval model からの出力を前処理してシリアライズすることをお勧めします。これにより、訓練時間を何時間も節約することができます。この前処理を実行しない場合、reader training パイプラインは、入力ファイルの拡張子が.pklであるかどうかをチェックし、そうでない場合は、前処理を行い、結果を元のファイルと同じフォルダに自動的にキャッシュします。

```bash
python preprocess_reader_data.py \
	--retriever_results {path to a file with results from dense_retriever.py} \
	--gold_passages {path to gold passages info} \
	--do_lower_case \
	--pretrained_model_cfg {pretrained_cfg} \
	--encoder_model_type {hf_bert | pytext_bert | fairseq_roberta} \
	--out_file {path to for output files} \
	--is_train_set
```



## Reader model training
```bash
python train_reader.py \
	--encoder_model_type {hf_bert | pytext_bert | fairseq_roberta} \
	--pretrained_model_cfg {bert-base-uncased| roberta-base} \
	--train_file "{globe expression for train files from #5 or #6 above}" \
	--dev_file "{globe expression for train files}" \
	--output_dir {path to output dir}
```

Notes:
- if you use pytext_bert or fairseq_roberta, you need to download pre-trained weights and specify --pretrained_file parameter. Specify the dir location of the downloaded files for 'pretrained.fairseq.roberta-base' resource prefix for RoBERTa model or the file path for pytext BERT (resource name 'pretrained.pytext.bert-base.model').
- Our current code only calculates __the Exact Match metric__.
- validation every `--eval_step batches`
- bi-encoder 同様，every validation で save model
- epoch 以外に stop condition はない
- Like the bi-encoder, there is no best checkpoint selection logic, so one needs to select that based on dev set validation performance which is logged in the train process output.
> 最適なチェックポイント選択ロジックは存在しないので、トレーニングプロセスの出力に記録されているデバイスセットの検証性能に基づいて選択する必要があります。

## Reader model inference

In order to make an inference, run `train_reader.py` without specifying `train_file`. Make sure to specify `model_file` with the path to the checkpoint, `passages_per_question_predict` with number of passages per question (being used when saving the prediction file), and `eval_top_docs` with a list of top passages threshold values from which to choose question's answer span (to be printed as logs). The example command line is as follows.
> 推論を行うには， train_file を指定せずに train_reader.py を実行します．model_file にはチェックポイントへのパスを、passages_per_question_predict には問題ごとの通過数を、eval_top_docs には問題の解答範囲を選択するためのトップ通過数のしきい値のリストを指定してください（ログとして出力されます）。

```bash
python train_reader.py \
  --prediction_results_file {some dir}/results.json \
  --eval_top_docs 10 20 40 50 80 100 \
  --dev_file {path to data.retriever_results.nq.single.test file} \
  --model_file {path to the reader checkpoint} \
  --dev_batch_size 80 \
  --passages_per_question_predict 100 \
  --sequence_length 350
```

## Distributed training
Use Pytorch's distributed training launcher tool:

```bash
python -m torch.distributed.launch \
	--nproc_per_node={WORLD_SIZE}  {non distributed scipt name & parameters}
```
Note:
- all batch size related parameters are specified per gpu in distributed mode(DistributedDataParallel) and for all available gpus in DataParallel (single node - multi gpu) mode.
> バッチサイズに関連するすべてのパラメータは、分散モード(DistributedDataParallel)ではgpuごとに、DataParallel(シングルノード-マルチgpu)モードでは利用可能なすべてのgpuに対して指定されます。


## Best hyperparameter settings

e2e example with the best settings for NQ dataset.

### 1. Download all retriever training and validation data:

```bash
python data/download_data.py --resource data.wikipedia_split.psgs_w100
python data/download_data.py --resource data.retriever.nq
python data/download_data.py --resource data.retriever.qas.nq
```

### 2. Biencoder(Retriever) training in single set mode.

We used distributed training mode on a single 8 GPU x 32 GB server

```bash
python -m torch.distributed.launch \
	--nproc_per_node=8 train_dense_encoder.py \
	--max_grad_norm 2.0 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--seed 12345 \
	--sequence_length 256 \
	--warmup_steps 1237 \
	--batch_size 16 \
	--do_lower_case \
	--train_file "{glob expression to train files downloaded as 'data.retriever.nq-train' resource}" \
	--dev_file "{glob expression to dev files downloaded as 'data.retriever.nq-dev' resource}" \
	--output_dir {your output dir} \
	--learning_rate 2e-05 \
	--num_train_epochs 40 \
	--dev_batch_size 16 \
	--val_av_rank_start_epoch 30
```
This takes about a day to complete the training for 40 epochs. It swiches to Average Rank validation on epoch 30 and it should be around 25 at the end.
The best checkpoint for bi-encoder is usually the last, but it should not be so different if you take any after epoch ~ 25.
> これで40エポック分のトレーニングが完了するまでに約1日かかります。エポック30で平均ランク検証にスウィッチして、最後は25くらいになるはずです。バイエンコーダーに最適なチェックポイントは大抵最後ですが、エポック～25以降はどれを取っても大差ないはずです。

### 3. Generate embeddings for Wikipedia.
Just use instructions for "Generating representations for large documents set". It takes about 40 minutes to produce 21 mln passages representation vectors on 50 2 GPU servers.
> 大規模な文書セットの表現を生成する」の指示を利用するだけです。50台のGPUサーバで2,100万パスの表現ベクトルを生成するのに約40分かかります。

### 4. Evaluate retrieval accuracy and generate top passage results for each of the train/dev/test datasets.

```bash
python dense_retriever.py \
	--model_file {path to checkpoint file from step 1} \
	--ctx_file {path to psgs_w100.tsv file} \
	--qa_file {path to test/dev qas file} \
	--encoded_ctx_file "{glob expression for generated files from step 3}" \
	--out_file {path for output json files} \
	--n-docs 100 \
	--validation_workers 32 \
	--batch_size 64
```

Adjust batch_size based on the available number of GPUs, 64 should work for 2 GPU server.

### 5. Reader training
We trained reader model for large datasets using a single 8 GPU x 32 GB server.

```bash
python train_reader.py \
	--seed 42 \
	--learning_rate 1e-5 \
	--eval_step 2000 \
	--do_lower_case \
	--eval_top_docs 50 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--train_file "{glob expression for train output files from step 4}" \
	--dev_file {glob expression for dev output file from step 4} \
	--warmup_steps 0 \
	--sequence_length 350 \
	--batch_size 16 \
	--passages_per_question 24 \
	--num_train_epochs 100000 \
	--dev_batch_size 72 \
	--passages_per_question_predict 50 \
	--output_dir {your save dir path}
```

We found that using the learning rate above works best with static schedule, so one needs to stop training manually based on evaluation performance dynamics.
Our best results were achieved on 16-18 training epochs or after ~60k model updates.
> 我々は、上記の学習率を使用することで、静的なスケジュールで最もよく動作することを発見しました。我々の最良の結果は、16-18回の学習エポック、または〜60k回のモデル更新後に達成されました。

We provide all input and intermediate results for e2e pipeline for NQ dataset and most of the similar resources for Trivia.
> e2e pipeline for NQ データセットのためのすべての入力結果と中間結果、およびトリビアのための同様のリソースのほとんどを提供しています。


## Misc.
- TREC validation requires regexp based matching. We support only retriever validation in regexp mode. See --math parameter options.
- WEbQ validation requires entity normalization, which is not included as of now.

## Reference

If you plan to use `DPR` in your project, please consider citing [our paper](https://arxiv.org/abs/2004.04906):
```
@misc{karpukhin2020dense,
    title={Dense Passage Retrieval for Open-Domain Question Answering},
    author={Vladimir Karpukhin and Barlas Oğuz and Sewon Min and Patrick Lewis and Ledell Wu and Sergey Edunov and Danqi Chen and Wen-tau Yih},
    year={2020},
    eprint={2004.04906},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License
DPR is CC-BY-NC 4.0 licensed as of now.


## Head Files

### datasets/retriever/qas/nq-dev.csv
```csv
who sings does he love me with reba	['Linda Davis']
how many pages is invisible man by ralph ellison	['581 ( second edition )']
where do the great lakes meet the ocean	['the Saint Lawrence River']
```

### datasets/gold_passages_info_nq_dev.json
```json
{
    "data": [
        {
            "question": "who sings does he love me with reba",
            "question_tokens": "who sings does he love me with reba",
            "context": "`` Does He Love You '' is a song written by Sandy Knox and Billy Stritch , and recorded as a duet by American country music artists Reba McEntire and Linda Davis . It was released in August 1993 as the first single from Reba 's album Greatest Hits Volume Two . It is one of country music 's several songs about a love triangle .",
            "example_id": -5263051561765837106,
            "short_answers": [
                "Linda Davis"
            ],
            "doc_url": "https://en.wikipedia.org//w/index.php?title=Does_He_Love_You&amp;oldid=847141247",
            "title": "Does He Love You"
        },
        {
            "question": "how many pages is invisible man by ralph ellison",
            "question_tokens": "how many pages is invisible man by ralph ellison",
            ...
```

### datasets/retriever/nq-dev.json
```json
[
    {
        "dataset": "nq_dev_psgs_w100",
        "question": "who sings does he love me with reba",
        "answers": [
            "Linda Davis"
        ],
        "positive_ctxs": [
            {
                "title": "Does He Love You",
                "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
                "score": 1000,
                "title_score": 1,
                "passage_id": "11828866"
            },
            {
                "title": "Does He Love You",
                "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
                "score": 13.394315,
                "title_score": 0,
                "passage_id": "11828866"
            },
            ...
```

### datasets/wikipedia_split/psgs_w100.tsv
```tsv
id	text	title
1	"Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from"	Aaron
```

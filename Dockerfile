FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        jq \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip install --no-cache-dir \
        "mecab-python3==0.996.5" \
        "pandas==1.1.4" \
        "tqdm" \
        "rank_bm25==0.2.1" \
        "transformers==2.11.0" \
        "spacy==2.3.2" \
        "faiss-cpu==1.6.3" \
        "tensorboard==2.3.0" \
        "torch==1.9.1"

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="cl-tohoku/bert-base-japanese-whole-word-masking"
RUN python -c "from transformers.modeling_bert import BertModel; BertModel.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
RUN python -c "from transformers import BertJapaneseTokenizer; BertJapaneseTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
ENV TRANSFORMERS_OFFLINE=1

# Copy files to the image
WORKDIR /app
COPY dpr dpr
COPY model model
COPY dense_retriever.py .
COPY train_reader.py .
COPY submission.sh .
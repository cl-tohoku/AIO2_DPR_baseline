# Copyright 2021 Masatoshi Suzuki (@singletongue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gzip
import json

from elasticsearch import Elasticsearch
from logzero import logger
from tqdm import tqdm


def create_index(es: Elasticsearch, index_name: str):
    es.indices.create(index=index_name, body={
        "settings": {
            "index": {
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "char_filter": [
                                "icu_normalizer"
                            ],
                            "tokenizer": "kuromoji_tokenizer",
                            "filter": [
                                "cjk_width",
                                "ja_stop",
                                "kuromoji_baseform",
                                "kuromoji_part_of_speech",
                                "kuromoji_stemmer",
                                "lowercase"
                            ]
                        }
                    }
                }
           }
        },
        "mappings": {
            "paragraph": {
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "paragraph_index": {"type": "integer"},
                    "text": {
                        "type": "text",
                        "analyzer": "custom_analyzer"
                    }
                }
            }
        }
    })


def index_passages(es: Elasticsearch, input_file: str, index_name: str):
    with gzip.open(input_file) as f:
        for line in tqdm(f):
            passage_item = json.loads(line)
            es.index(index=index_name, doc_type="paragraph", body={
                "id": passage_item["id"],
                "title": passage_item["title"],
                "text": passage_item["text"]
            })


def main(args):
    es = Elasticsearch(hosts=[{"host": args.hostname, "port": args.port}], timeout=60)

    logger.info("Creating an Elasticsearch index")
    create_index(es, args.index_name)

    logger.info("Indexing documents")
    index_passages(es, args.input_file, args.index_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--index_name", type=str, required=True)
    parser.add_argument("--hostname", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    args = parser.parse_args()
    main(args)

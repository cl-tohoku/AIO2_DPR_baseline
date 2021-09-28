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
from tqdm import tqdm


class ElasticsearchPassageRetriever():
    def __init__(self, index, host, port):
        self.es = Elasticsearch(hosts=[{"host": host, "port": port}], timeout=60)
        self.index = index

    def query(self, query_text, size=10):
        query = {
            "query": {
                "match": {
                    "text": query_text
                }
            },
            "size": size
        }
        response = self.es.search(index=self.index, body=query)
        documents = []
        for item in response["hits"]["hits"]:
            document = {
                "id": item["_source"]["id"],
                "title": item["_source"]["title"],
                "text": item["_source"]["text"]
            }
            documents.append(document)

        return documents

    def query_with_filtering(self, query_text, filter_text, filter_is_must_not=False, size=10):
        filter_key = "must_not" if filter_is_must_not else "filter"
        query = {
            "query": {
                "bool": {
                    "must": {
                        "match": {
                            "text": query_text
                        }
                    },
                    filter_key: {
                        "match_phrase": {
                            "text": filter_text
                        }
                    }
                }
            },
            "size": size
        }
        response = self.es.search(index=self.index, body=query)
        documents = []
        for item in response["hits"]["hits"]:
            title = item["_source"]["title"]
            text = item["_source"]["text"]

            if filter_is_must_not and filter_text in text:
                continue
            if not filter_is_must_not and filter_text not in text:
                continue

            document = {
                "id": item["_source"]["id"],
                "title": title,
                "text": text
            }
            documents.append(document)

        return documents


def main(args):
    passage_retriever = ElasticsearchPassageRetriever(
        index=args.es_index_name, host=args.es_hostname, port=args.es_port
    )
    num_positive_question = 0
    num_negative_question = 0

    output_items = []
    with open(args.input_file) as f:
        for line in tqdm(f):
            item = json.loads(line)

            passages = passage_retriever.query(item["question"], size=args.num_documents_per_question)

            positive_passages = []
            negative_passages = []
            for passage in passages:
                for answer in item["answers"]:
                    if answer in passage["text"]:
                        positive_passages.append(passage)
                        break
                else:
                    negative_passages.append(passage)

            if len(positive_passages) > 0:
                num_positive_question += 1
            else:
                num_negative_question += 1

            output_item = {
                "qid": item["qid"],
                "timestamp": item["timestamp"],
                "question": item["question"],
                "answers": item["answers"],
                "positive_ctxs": positive_passages,
                "negative_ctxs": [],
                "hard_negative_ctxs": negative_passages,
            }
            output_items.append(output_item)

    assert num_positive_question + num_negative_question == len(output_items)
    print("Questions with at least one positive document:", num_positive_question)
    print("Questions with no positive document:", num_negative_question)
    print("Total output questions:", len(output_items))

    with gzip.open(args.output_file, "wt") as fo:
        json.dump(output_items, fo, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_documents_per_question", type=int, default=100)
    parser.add_argument("--es_index_name", type=str, required=True)
    parser.add_argument("--es_hostname", type=str, default="localhost")
    parser.add_argument("--es_port", type=int, default=9200)
    args = parser.parse_args()
    main(args)

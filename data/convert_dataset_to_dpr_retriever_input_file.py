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

from tqdm import tqdm


def main(args: argparse.Namespace):
    with gzip.open(args.input_file, "rt") as f, gzip.open(args.output_file, "wt") as fo:
        if not args.write_jsonl:
            fo.write("[")

        for i, line in tqdm(enumerate(f)):
            qa_item = json.loads(line)

            documents = qa_item["documents"]
            positive_documents = [documents[i] for i in qa_item["positive_document_indices"]]
            negative_documents = [documents[i] for i in qa_item["negative_document_indices"]]

            output_item = {
                "qid": qa_item["qid"],
                "timestamp": qa_item["timestamp"],
                "question": qa_item["question"],
                "answers": [qa_item["answer"]],
                "positive_ctxs": positive_documents,
                "negative_ctxs": [],
                "hard_negative_ctxs": negative_documents
            }
            if args.write_jsonl:
                print(json.dumps(output_item, ensure_ascii=False), file=fo)
            else:
                if i > 0:
                    fo.write(",")
                for line in json.dumps(output_item, ensure_ascii=False, indent=4).split("\n"):
                    fo.write("\n    " + line)

        if not args.write_jsonl:
            fo.write("\n]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--write_jsonl", action="store_true")
    args = parser.parse_args()
    main(args)

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


def generate_passages(paragraphs_filename: str, max_passage_length: int):
    passage_id = 0
    last_title = ""
    last_section = ""
    passage_texts = []

    with gzip.open(paragraphs_filename, "rt") as f:
        for line in f:
            paragraph_item = json.loads(line)
            title = paragraph_item["title"]
            section = paragraph_item["section"]
            paragraph_text = paragraph_item["text"]

            if section != last_section:
                for passage_text in passage_texts:
                    passage_id += 1
                    yield dict(id=passage_id, title=last_title, text=passage_text)

                passage_texts = []

            if len(paragraph_text) <= max_passage_length:
                passage_texts.append(paragraph_text)

            last_title = title
            last_section = section
        else:
            for passage_text in passage_texts:
                passage_id += 1
                yield dict(id=passage_id, title=last_title, text=passage_text)


def main(args):
    with gzip.open(args.output_file, "wt") as fo:
        passage_generator = generate_passages(
            paragraphs_filename=args.input_file,
            max_passage_length=args.max_passage_length,
        )
        for passage_item in tqdm(passage_generator):
            print(json.dumps(passage_item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_passage_length", type=int, default=1000)
    args = parser.parse_args()
    main(args)

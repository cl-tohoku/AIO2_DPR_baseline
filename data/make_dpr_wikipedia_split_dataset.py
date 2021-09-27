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


HEADER_COLS = ["id", "text", "title"]


def main(args: argparse.Namespace):
    with gzip.open(args.input_file, "rt") as f, gzip.open(args.output_file, "wt") as fo:
        print(*HEADER_COLS, sep="\t", file=fo)
        for line in tqdm(f):
            paragraph_item = json.loads(line)

            id_ = paragraph_item["id"]

            title = paragraph_item["title"]
            assert "\t" not in title

            text = paragraph_item["text"]
            text = json.dumps(text, ensure_ascii=False)
            assert "\t" not in text

            print(id_, text, title, sep="\t", file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)

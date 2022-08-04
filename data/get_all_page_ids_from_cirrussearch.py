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


def main(args):
    with gzip.open(args.cirrus_file, "rt") as f, open(args.output_file, "w") as fo:
        title = None
        page_id = None
        rev_id = None
        for line in tqdm(f):
            item = json.loads(line)
            if "index" in item:
                page_id = item["index"]["_id"]
            else:
                assert page_id is not None

                title = item["title"]
                rev_id = item["version"]
                templates = item["template"]
                num_inlinks = item.get("incoming_links", 0)

                if args.min_inlinks is not None and num_inlinks < args.min_inlinks:
                    continue
                if args.exclude_disambiguation_pages and "Template:Dmbox" in templates:
                    continue
                if args.exclude_sexual_pages and "Template:性的" in templates:
                    continue
                if args.exclude_violent_pages and "Template:暴力的" in templates:
                    continue

                output_item = {
                    "title": title,
                    "pageid": page_id,
                    "revid": rev_id
                }
                print(json.dumps(output_item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cirrus_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--min_inlinks", type=int)
    parser.add_argument("--exclude_disambiguation_pages", action="store_true")
    parser.add_argument("--exclude_sexual_pages", action="store_true")
    parser.add_argument("--exclude_violent_pages", action="store_true")
    args = parser.parse_args()
    main(args)

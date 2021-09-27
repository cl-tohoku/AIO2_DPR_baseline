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
from time import sleep

import grequests
import requests
from logzero import logger
from tqdm import tqdm


def handle_request_exception(request, exception):
    logger.warning(f'request "{request.url}" failed: {exception}')


def main(args):
    assert args.batch_size < 200, "batch_size should be limited to no more than 200."

    base_url = "https://ja.wikipedia.org/api/rest_v1"

    page_items = []

    logger.info("Loading the Page IDs file")
    with open(args.page_ids_file) as f:
        for line in f:
            input_item = json.loads(line)
            title = input_item["title"]
            page_id = input_item["pageid"]
            rev_id = input_item["revid"]

            url = "{}/page/mobile-html/{}/{}".format(base_url, title.replace(" ", "_"), rev_id)

            page_items.append((title, page_id, rev_id, url))

    failed_pages = []

    logger.info("Retrieving Page HTMLs")
    with gzip.open(args.output_file, "wt") as fo, tqdm(total=len(page_items)) as pbar:
        i = 0
        while i < len(page_items):
            batch_pages = page_items[i:i + args.batch_size]
            assert len(batch_pages) > 0

            reqs = (grequests.get(page_item[3], timeout=60) for page_item in batch_pages)
            responses = grequests.map(reqs, exception_handler=handle_request_exception)
            assert len(responses) == len(batch_pages)

            for page_item, response in zip(batch_pages, responses):
                (title, page_id, rev_id, url) = page_item
                if response is not None:
                    output_item = {
                        "title": title,
                        "pageid": page_id,
                        "revid": rev_id,
                        "url": url,
                        "html": response.text
                    }
                    print(json.dumps(output_item, ensure_ascii=False), file=fo)
                    pbar.update(1)
                else:
                    failed_pages.append(page_item)

            i += args.batch_size

            # Requests should be limited to 200 requests/sec.
            # See https://en.wikipedia.org/api/rest_v1/.
            sleep(args.batch_size / 200)

        if len(failed_pages) > 0:
            logger.info("Retrying failed %s requests", len(failed_pages))
            for page_item in failed_pages:
                (title, page_id, rev_id, url) = page_item
                try:
                    response = requests.get(url, timeout=300)
                    response.raise_for_status()
                except Exception as exception:
                    logger.warning("Request for %s failed: %s", url, exception)
                    continue

                output_item = {
                    "title": title,
                    "pageid": page_id,
                    "revid": rev_id,
                    "url": url,
                    "html": response.text
                }
                print(json.dumps(output_item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--page_ids_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()
    main(args)

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import os

TARGET_DATA_FOLDER = "./raw_data/"


def main():
    with open("./datagetconfig.json", "r") as fp:
        confdata = json.load(fp)

    for idx, datum in enumerate(confdata):
        url = datum["url"]
        text = get_text_from_url(url)

        lines = text.split("\n")
        linecount = len(lines)
        print(f"Retrieved {linecount} lines of text.")

        save_filepath = os.path.join(TARGET_DATA_FOLDER, f"{idx:03d}.txt")
        with open(save_filepath, "w") as fp:
            fp.write(text)


def get_text_from_url(url):
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1)
    s.mount("https://", HTTPAdapter(max_retries=retries))

    response = s.get(url=url)

    assert response.status_code == 200, "Failed to retrieve data."

    response.encoding = response.apparent_encoding

    return response.text


if __name__ == "__main__":
    main()

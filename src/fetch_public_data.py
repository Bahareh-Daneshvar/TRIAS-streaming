"""Download the public Data 1 files named in the paper.

Run in a normal internet-enabled environment:
    python src/fetch_public_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/wwwxmu/Dataset-of-financial-news-sentiment-classification/master"
FILES = ["train_data.csv", "test_data.csv", "news_seed.xlsx"]


def main():
    out = Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = f"{BASE}/{name}"
        target = out / name
        print(f"Downloading {url} -> {target}")
        urlretrieve(url, target)
    print("Done.")


if __name__ == "__main__":
    main()

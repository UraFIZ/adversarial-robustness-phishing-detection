"""Завантаження датасету Vrbancic et al. (фішингові URL) у data/.
Датасет не зберігається в репозиторії; це джерело відтворюваності."""
import os, urllib.request

URL = ("https://raw.githubusercontent.com/GregaVrbancic/"
       "Phishing-Dataset/master/dataset_small.csv")
DST = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_small.csv")

def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    print(f"Завантаження {URL} ...")
    urllib.request.urlretrieve(URL, DST)
    size = os.path.getsize(DST) / 1e6
    print(f"Збережено: {os.path.abspath(DST)} ({size:.1f} МБ)")

if __name__ == "__main__":
    main()

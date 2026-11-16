"""Download the two raw CSV files from Kaggle into data/raw/ (owner: R1).

Needs a Kaggle API token in ~/.kaggle/kaggle.json (https://www.kaggle.com/docs/api).
Without a token, download RAW_recipes.csv and RAW_interactions.csv manually from
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
and put them in data/raw/.
"""
from foodcom import config

FILES = ["RAW_recipes.csv", "RAW_interactions.csv"]


def download(force: bool = False) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    for name in FILES:
        if (config.DATA_RAW / name).exists() and not force:
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading…")
        api.dataset_download_file(config.KAGGLE_DATASET, name, path=config.DATA_RAW, force=force)
        _unzip_if_needed(name)


def _unzip_if_needed(name: str) -> None:
    import zipfile
    archive = config.DATA_RAW / f"{name}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(config.DATA_RAW)
        archive.unlink()


if __name__ == "__main__":
    download()

from fire import Fire

from config_make import get_config
from trainer import Trainer
from utils import *


def train(config=None):
    if config is None:
        config = get_config()

    prepare_dirs(config)

    if config.resume:
        try:
            save_config(config)
        except ValueError:
            print("[!] config data already exist. Either change the model number, or delete the json file and rerun.")
            return

    trainer = Trainer(config)
    trainer.train()


def test(config=None):
    if config is None:
        config = get_config()


def run():
    config = get_config()
    train(config)
    test()


def download_data():
    download_data()


if __name__ == '__main__':
    # config = get_config()
    Fire({"run": run, "download-data": download_data, "train": train, })

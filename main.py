from fire import Fire

from config_make import get_config
from trainer import Trainer
from utils import *


def print_status(string):
    line = '*' * 40
    print(line + " " + string + " " + line)


# only train and validation
def train(config=None, trainer=None):
    if config or trainer is None:
        config = get_config()
        trainer = Trainer(config)

    # Make directory for save logs and model
    prepare_dirs(config)

    # Check resume data
    if config.resume:
        try:
            print(f"load saved config data of model number {config.num_model}")
            save_config(config)
        except ValueError:
            print("[!] config data already exist. Either change the model number, or delete the json file and rerun.")
            return

    # train model
    print_status("Train Start")
    trainer.train()


# only test
def test(config=None, trainer=None):
    if config or trainer is None:
        config = get_config()
        trainer = Trainer(config)

    # test model
    print_status("Test Start")
    trainer.test()


# running all process. download data, data set, data loader, train, validation, test
def run():
    download_omniglot_data()

    # Make options
    config = get_config()

    # Make Trainer
    trainer = Trainer(config)

    # train
    train(config, trainer)

    # test
    test(config, trainer)


def download_data():
    download_omniglot_data()


if __name__ == '__main__':
    Fire({"run": run, "download-data": download_data, "train": train, "test": test})

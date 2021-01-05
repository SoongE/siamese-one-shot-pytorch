from fire import Fire

from utils import *
from config import get_config

from pprint import pprint as pp


def run():
    # print(Config.test)
    # print(Config.background_path)
    # print(Config.evaluation_path)
    pass


def download_data():
    download_data()


def test():
    config, config2, config3 = get_config()
    pp(config.way)

    pp(config2)

    pp(config3)


if __name__ == '__main__':
    # Fire({"run": run, "download-data": download_data})
    test()

import json
import os
import shutil
from glob import glob
from zipfile import ZipFile

import wget


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prepare_dirs(config):
    for path in [config.ckpt_dir, config.logs_dir, config.plot_dir]:
        path = os.path.join(path, config.num_model)
        if not os.path.exists(path):
            os.makedirs(path)
        if config.flush:
            shutil.rmtree(path)
            if not os.path.exists(path):
                os.makedirs(path)


def save_config(config):
    model_dir = os.path.join(config.ckpt_dir, config.num_model)
    param_path = os.path.join(model_dir, 'params.json')

    if not os.path.isfile(param_path):
        print(f"Save params in {param_path}")

        all_params = config.__dict__
        with open(param_path, 'w') as fp:
            json.dump(all_params, fp, indent=4, sort_keys=True)
    else:
        raise ValueError


# download omniglot dataset
def download_omniglot_data():
    BASEDIR = os.path.dirname(os.path.realpath(__file__)) + '/data'

    if not os.path.exists(BASEDIR):
        os.mkdir(BASEDIR)

    if not os.path.exists(BASEDIR + '/raw/images_background.zip'):
        print("download background image")
        wget.download("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip",
                      BASEDIR + '/raw')
    if not os.path.exists(BASEDIR + '/raw/images_evaluation.zip'):
        print("download evaluation image")
        wget.download("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip",
                      BASEDIR + '/raw')

    for d in glob(BASEDIR + '/raw/*.zip'):
        zip_name = os.path.splitext(os.path.basename(d))[0]
        print(f'{zip_name}is being unzipped...', end="")
        with ZipFile(d, 'r') as zip_object:
            zip_object.extractall(BASEDIR + '/unzip/')
        print("success")

    try:
        os.rename(BASEDIR + '/unzip/images_background', BASEDIR + '/unzip/background')
        os.rename(BASEDIR + '/unzip/images_evaluation', BASEDIR + '/unzip/evaluation')
    except FileNotFoundError as e:
        print(e)

    print("DONE.")


if __name__ == '__main__':
    download_data()

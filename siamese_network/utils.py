import json
import os
import shutil
from glob import glob
from zipfile import ZipFile

import wget
from prettytable import PrettyTable


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
    path = config.logs_dir
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, 'logs'))
        os.makedirs(os.path.join(path, 'models'))
    if config.flush:
        shutil.rmtree(path)
        os.makedirs(os.path.join(path, 'logs'))
        os.makedirs(os.path.join(path, 'models'))


def save_config(config):
    param_path = os.path.join(config.logs_dir, 'params.json')

    if not os.path.isfile(param_path):
        print(f"Save params in {param_path}")

        all_params = config.__dict__
        with open(param_path, 'w') as fp:
            json.dump(all_params, fp, indent=4, sort_keys=True)
    else:
        print(f"[!] Config file already exist.")
        raise ValueError


def load_config(config):
    param_path = os.path.join(config.logs_dir, 'params.json')
    params = json.load(open(param_path))

    config.__dict__.update(params)

    config.resume = True

    return config


# download omniglot dataset
def download_omniglot_data():
    BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)) + '/data'

    # make directory
    if not os.path.exists(BASEDIR):
        os.mkdir(BASEDIR)
    if not os.path.exists(os.path.join(BASEDIR, 'unzip')):
        os.mkdir(os.path.join(BASEDIR, 'unzip'))

    # download zip file
    if not os.path.exists(BASEDIR + '/raw/images_background.zip'):
        print("download background image")
        wget.download("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip",
                      BASEDIR + '/raw')
    if not os.path.exists(BASEDIR + '/raw/images_evaluation.zip'):
        print("download evaluation image")
        wget.download("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip",
                      BASEDIR + '/raw')

    # if there are no unzipped files
    if not any([True for _ in os.scandir(os.path.join(BASEDIR, "unzip"))]):
        # unzip files
        for d in glob(BASEDIR + '/raw/*.zip'):
            zip_name = os.path.splitext(os.path.basename(d))[0]
            print(f'{zip_name}is being unzipped...', end="")
            with ZipFile(d, 'r') as zip_object:
                zip_object.extractall(BASEDIR + '/unzip/')
            print("success")

        # change folder name
        try:
            os.rename(BASEDIR + '/unzip/images_background', BASEDIR + '/unzip/background')
            os.rename(BASEDIR + '/unzip/images_evaluation', BASEDIR + '/unzip/evaluation')
        except FileNotFoundError as e:
            print(e)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

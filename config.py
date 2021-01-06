import argparse

import torch
from pprint import pprint as pp


def str2bool(v):
    return v.lower() in ('False', '1')


parser = argparse.ArgumentParser(description='Siamese Network')

# data params
data_arg = parser.add_argument_group('Data Params')
data_arg.add_argument('--way', type=int, default=20, help='Ways in the 1-shot trials')
data_arg.add_argument('--num_train', type=int, default=300, help='# of images in train dataset')
data_arg.add_argument('--batch_size', type=int, default=64, help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=1, help='# of subprocesses to use for data loading')
# other params
misc_arg = parser.add_argument_group('other Params')
misc_arg.add_argument('--data_dir', type=str, default='./data/changed/', help='Directory in which data is stored')


def get_config():
    config = parser.parse_args()
    return config


if __name__ == '__main__':
    config = get_config()

    if torch.cuda.is_available():
        config.num_workers = 1

    pp(config)

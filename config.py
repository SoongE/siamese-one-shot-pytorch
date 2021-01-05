import argparse


def str2bool(v):
    return v.lower() in ('False', '1')


parser = argparse.ArgumentParser(description='Siamese Network')

# data params
data_arg = parser.add_argument_group('Data Params')
data_arg.add_argument('--way', type=int, default=20, help='Ways in the 1-shot trials')
data_arg.add_argument('--num_train', type=int, default=300, help='# of images in train dataset')
data_arg.add_argument('--batch_size', type=int, default=64, help='# of images in each batch of data')

# other params
misc_arg = parser.add_argument_group('other Params')
misc_arg.add_argument('--data_dir', type=str, default='./data/changed/', help='Directory in which data is stored')


def get_config():
    config = parser.parse_args()
    return config

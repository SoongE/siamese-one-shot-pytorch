import os
import wget
from zipfile import ZipFile
from glob import glob


# download omniglot dataset
def download_data():
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

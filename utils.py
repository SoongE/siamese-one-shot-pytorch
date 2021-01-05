import os
import wget
from zipfile import ZipFile
from glob import glob


# download omniglot dataset
def download_data():
    BASEDIR = os.getcwd() + '/data'

    if not os.path.exists(BASEDIR):
        os.mkdir(BASEDIR)

    if not os.path.exists(BASEDIR + '/images_background.zip'):
        wget.download("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip",
                      BASEDIR)
    if not os.path.exists(BASEDIR + '/images_evaluation.zip'):
        wget.download("https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip",
                      BASEDIR)

    for d in glob(BASEDIR + '/*.zip'):
        zip_name = os.path.splitext(os.path.basename(d))[0]
        print(f'{zip_name}is being unzipped...', end="")
        with ZipFile(d, 'r') as zip_object:
            zip_object.extractall(BASEDIR + '/processed/')
        print("success")

    try:
        os.rename(BASEDIR + '/processed/images_background', BASEDIR + '/processed/background')
        os.rename(BASEDIR + '/processed/images_evaluation', BASEDIR + '/processed/evaluation')
    except FileNotFoundError as e:
        print(e)

    print("DONE.")

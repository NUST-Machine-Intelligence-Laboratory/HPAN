import os

from easydict import EasyDict

OPTION = EasyDict()
OPTION.ROOT_PATH = os.path.join(os.path.expanduser('.'))
OPTION.SNAPSHOTS_DIR = os.path.join(OPTION.ROOT_PATH, 'workdir')
OPTION.DATASETS_DIR = os.path.join(OPTION.ROOT_PATH, 'data')

OPTION.TRAIN_SIZE = (241, 425)
OPTION.TEST_SIZE = (241, 425)

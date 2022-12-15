from os.path import join

from dataset import DatasetFromFolder_Test


def get_test_set(root_dir):
    test_dir = join(root_dir)

    return DatasetFromFolder_Test(test_dir)

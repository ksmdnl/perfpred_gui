import os

from dataloader.file_io import dataset_index
from dataloader.file_io import filelist_creator, split_creator
# dataset_index.create_parameter_files("cityscapes")
path = os.getenv("IFN_DIR_DATASET")
# dataset = "cityscapes"
dataset = "cityscapes"

data_creator = filelist_creator.DatasetCreator(dataset, rewrite=True)
check = data_creator.check_state()
if not check:
    data_creator.create_dataset()


dataset = 'cityscapes'
splits = ['train', 'validation', 'test']
splits = None

data_splitter = split_creator.DatasetSplitter(dataset, splits)
data_splitter.create_splits()

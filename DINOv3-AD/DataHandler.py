from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random

from Datasets import DATASET_REGISTRY

class DataSampler():
    # def __init__(self, dataset_name: str, class_name: str):
    #     self.dataset_class, self.split_class, self.root_path = DATASET_REGISTRY[dataset_name]
    #     self.dataset = self.dataset_class(
    #         root=self.root_path,
    #         split=self.split_class.TEST,
    #         class_name=class_name,
    #         transform=None,
    #     )


    ''' Create a dataloader for a specific category from the dataset registry '''
    def prepare_data_from_registry(dataset_name, category, batch_size, device, resize=512, imagesize=512, **kwargs):
        dataset_name = dataset_name.lower()
        if DATASET_REGISTRY is None:
            raise RuntimeError("DATASET_REGISTRY not available. Provide images differently.")
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
        test_dataset = dataset_cls(source=root_path, split=split_cls.TEST, classname=category, resize=resize, imagesize=imagesize)
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        return loader
    

    ''' Get a random sample from a specific category from the dataset registry '''
    def get_sample_from_registry(dataset_name, category, resize=512, imagesize=512, **kwargs):
        dataset_name = dataset_name.lower()
        if DATASET_REGISTRY is None:
            raise RuntimeError("DATASET_REGISTRY not available. Provide images differently.")
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
        test_dataset = dataset_cls(source=root_path, split=split_cls.TEST, classname=category, resize=resize, imagesize=imagesize)
        idx = random.randint(0, len(test_dataset) - 1)
        sample = test_dataset[idx]   # returns dict
        return sample
    
    
    def get_all_good_samples_from_category(dataset_name, category, resize=512, imagesize=512, **kwargs):
        dataset_name = dataset_name.lower()
        if DATASET_REGISTRY is None:
            raise RuntimeError("DATASET_REGISTRY not available. Provide images differently.")
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        dataset_cls, split_cls, root_path = DATASET_REGISTRY[dataset_name]
        good_dataset = dataset_cls(source=root_path, split=split_cls.TRAIN, classname=category, resize=resize, imagesize=imagesize)
        
        good_samples = []
        for idx in range(len(good_dataset)):
            sample = good_dataset[idx]
            good_samples.append(sample)
        
        print(f"Found {len(good_samples)} good samples in category '{category}' of dataset '{dataset_name}'.")
        
        return good_samples
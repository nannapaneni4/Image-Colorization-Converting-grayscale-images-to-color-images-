from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from PIL import Image
from torchvision.transforms.functional import resize

class ColorizeData(Dataset):
    def __init__(self,df):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.df=df
        #self.clean_dataset()
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def clean_dataset(self):
        for idx, row in self.df.iterrows():
            file = row['file_name']
            if 'jpg' not in file:
                print("{} is a invalid file".format(file))
                self.df.drop(idx, inplace=True)
                continue
            img = Image.open(file)
            if T.ToTensor()(img).shape[0] != 3:
                print("{} is a GrayScale Image".format(file))
                self.df.drop(idx, inplace=True)
    
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.df)
        pass
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        file = self.df.iloc[index]['file_name']
        img = Image.open(file)
        ip_trans = self.input_transform(img)
        targ_trans = self.target_transform(img)
        return (ip_trans, targ_trans)
        pass
        
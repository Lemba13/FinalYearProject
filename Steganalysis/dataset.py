import numpy as np
import pandas as pd
import torch
import cv2
import config
from torch.utils import data

from torch.utils.data import Dataset,DataLoader

class StegDataset(Dataset):
    def __init__(self, csv_file, norm=False, transform = None, gray=False, resize = False):
        self.filenames = pd.read_csv(csv_file)
        self.transform = transform
        self.norm = norm
        self.gray = gray
        self.resize = resize
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        img_path = 'detection_dataset/'+self.filenames.iloc[index, 0]
        if self.gray == False:
            img = cv2.imread(img_path)
            s=3
        else:    
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            s=1
        
        label = self.filenames.iloc[index,1]
        
        if self.resize:
            img = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)
            
        img = np.reshape(img, (img.shape[0], img.shape[1], s))
        
        if self.norm == True:
            img = img/255.0
            
        if self.transform:
            transformation = self.transform(image=img)
            img = transformation['image']
            
        return img.type(torch.FloatTensor), label
    
    

    
def test():
    dataset = StegDataset(
        'csv_files/final_steganalysis_6000_samples.csv',
        norm = False,
        transform=config.transform
    )
    
    sampleloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    print(next(iter(sampleloader))[0].shape)
    print(next(iter(sampleloader))[0])


if __name__ == '__main__':
    test()

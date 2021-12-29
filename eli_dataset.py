import numpy as np
import pandas as pd
import torch
import cv2
import config
from torch.utils.data import Dataset, DataLoader

class EliDataset(Dataset):
    def __init__(self, csv_file, transform =  None):
        self.filenames = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        cover_path = 'detection_dataset/'+self.filenames.iloc[index, 0]
        encoded_path = 'detection_dataset/'+self.filenames.iloc[index, 1]
        
        cover_img = cv2.imread(cover_path)
        encoded_img = cv2.imread(encoded_path)
        
        if self.transform:
            transformation1 = self.transform(image= cover_img)
            transformation2 = self.transform(image = encoded_img)
            cover_img = transformation1['image']
            encoded_img = transformation2['image']
            
        return cover_img.type(torch.FloatTensor), encoded_img.type(torch.FloatTensor)
    
    
def test():
    dataset = EliDataset(
        'csv_files/elimination_1000_samples.csv',
        transform=config.transform
    )
    
    sampleloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    print(next(iter(sampleloader))[0].shape)
    print(next(iter(sampleloader))[0])


if __name__ == '__main__':
    test()     

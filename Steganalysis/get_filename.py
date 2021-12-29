import os
import pandas as pd
import random
from PIL import Image
import shutil
from tqdm import tqdm

filenames_all = []
for item in os.listdir('Dataset/Cover'):
    filenames_all.append(item.split('.')[0])
    
n = int(input('Number of samples(steganalysis):'))

filenames = random.sample(filenames_all,n)

df = pd.DataFrame(columns=['path', 'label'])

for i in range(n):
    cover_path = 'Dataset/Cover/'
    content_path = random.choice(['Dataset/JMiPOD/', 'Dataset/JUNIWARD/', 'Dataset/UERD/'])
    cover = {'path': cover_path+filenames[i]+'.jpg', 'label': 0}
    content = {'path': content_path+filenames[i]+'.jpg', 'label': 1}
    df = df.append(cover, ignore_index=True)
    df = df.append(content, ignore_index=True)
    
df = df.sample(frac=1).reset_index(drop=True)

try:
    os.mkdir('detection_dataset')
except FileExistsError:
    shutil.rmtree('detection_dataset')
    
os.mkdir('detection_dataset')


final_df = pd.DataFrame()

for i in tqdm(range(len(df))):
    img = Image.open(df.path[i])
    filename = df.path[i].split('/')[1]+'_'+df.path[i].split('/')[2]
    data = {'path': filename, 'label': df.label[i]}
    img.save('detection_dataset/'+filename)
    final_df = final_df.append(data, ignore_index=True)

final_df = final_df.sample(frac=1).reset_index(drop=True)
final_df.to_csv('csv_files/final_steganalysis_'+str(n)+'_samples.csv', index=False)

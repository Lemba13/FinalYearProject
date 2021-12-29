import os
import pandas as pd
import random
from tqdm import tqdm

df = pd.read_csv('csv_files/final_steganalysis_6000_samples.csv')

ids = []
for i in (range(len(df))):
    id = df.path[i].split('_')[1].split('.jpg')[0]
    if id not in ids:
        ids.append(id)

n = int(input('Number of samples(detection):'))

sampled_id = random.sample(ids, n)

df0 = pd.DataFrame()
for i in tqdm(range(len(sampled_id))):
    cover_img = 'Cover_'+sampled_id[i]+'.jpg'
    if 'JMiPOD_'+sampled_id[i]+'.jpg' in os.listdir('detection_dataset'):
        encoded_img = 'JMiPOD_'+sampled_id[i]+'.jpg'
    elif 'JUNIWARD_'+sampled_id[i]+'.jpg' in os.listdir('detection_dataset'):
        encoded_img = 'JUNIWARD_'+sampled_id[i]+'.jpg'
    else:
        encoded_img = 'UERD_'+sampled_id[i]+'.jpg'
    temp_dict = {'cover': cover_img, 'encoded': encoded_img}
    df0 = df0.append(temp_dict, ignore_index=True)
    
df0.to_csv('csv_files/elimination_'+str(n)+'_samples.csv', index=False)

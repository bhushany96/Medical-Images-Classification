import numpy as np
import pandas as pd
from glob import glob
import os
from sklearn.model_selection import train_test_split
import shutil

train_n = 10000
test_n = 3200
train_b = int(np.ceil(10000/14))
test_b = int(np.ceil(3200/14))


all_xray_df = pd.read_csv('./Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('.', 'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

clean_df = all_xray_df[all_xray_df['Finding Labels'] != 'No Finding']
clean_df["Finding Labels"] = clean_df['Finding Labels'].map(lambda x: x if '|' not in x else x.split('|'))
clean_df = clean_df.explode('Finding Labels')

labels = np.unique(clean_df['Finding Labels']).tolist()

split_df = clean_df.groupby('Finding Labels') 

train_df = pd.DataFrame(columns = list(clean_df.columns))
test_df = pd.DataFrame(columns = list(clean_df.columns))
for l in labels:
    tr, ts = train_test_split(split_df.get_group(l), test_size = 0.25, random_state = 1, shuffle = True)
    train_df = train_df.append(tr)
    test_df = test_df.append(ts)

train_df.to_csv('train_df.csv')
test_df.to_csv('test_df.csv')

train_folder_df = train_df.groupby('Finding Labels')
test_folder_df = test_df.groupby('Finding Labels')

train_path = './data/train'
test_path = './data/test'

os.mkdir('./data')
os.mkdir(train_path)
os.mkdir(test_path)

for l in labels:
	tr = os.path.join(train_path, l)
	ts = os.path.join(test_path, l)
	os.mkdir(tr)
	os.mkdir(ts)
	
for l in labels:
	c = 0
	for p in train_folder_df.get_group(l)['path']:
		if c != train_b:
			shutil.copy(p, train_path + '/' + l)
			c += 1
		else:
			break

for l in labels:
	c = 0	
	for p in test_folder_df.get_group(l)['path']:
		if c!= test_b:
			shutil.copy(p, test_path + '/' + l)
			c += 1
		else:
			break

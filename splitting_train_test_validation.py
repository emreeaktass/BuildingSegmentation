from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
import file_operations as f

def train_test_validation_split_():
    path_data = 'all_datas/data/'
    path_mask = 'all_datas/mask/'
    file_names_data = glob(path_data + '*')
    file_names_mask = glob(path_mask + '*')
    file_names_data.sort()
    file_names_mask.sort()
    datas = file_names_data
    labels = file_names_mask
    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.1, random_state=47)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=47)

    t_path = 'dataset/test'
    v_path = 'dataset/validation'
    tr_path = 'dataset/train'

    for i, j in zip(X_test, y_test):
        f.write(t_path, i.split('/')[-1].split('.')[0], 'data', np.load(i))
        f.write(t_path, j.split('/')[-1].split('.')[0], 'mask', np.load(j))

    for i, j in zip(X_val, y_val):
        f.write(v_path, i.split('/')[-1].split('.')[0], 'data', np.load(i))
        f.write(v_path, j.split('/')[-1].split('.')[0], 'mask', np.load(j))

    for i, j in zip(X_train, y_train):
        f.write(tr_path, i.split('/')[-1].split('.')[0], 'data', np.load(i))
        f.write(tr_path, j.split('/')[-1].split('.')[0], 'mask', np.load(j))

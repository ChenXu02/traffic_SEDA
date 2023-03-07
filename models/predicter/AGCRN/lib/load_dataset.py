import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data_path_missing = os.path.join('../data/PeMSD8/pems08_pm_random_mat_40%25.npy')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
        data_missing = np.load(data_path_missing)[:, 0, :].transpose(1, 0)
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data_path_missing = os.path.join('../data/PeMSD8/pems08_pm_random_mat_40%25.npy')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
        data_missing = np.load(data_path_missing)[:,0,:].transpose(1,0)  # onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        data_missing = np.expand_dims(data_missing, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data,data_missing

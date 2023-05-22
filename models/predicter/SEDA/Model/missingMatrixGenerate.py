import numpy as np
import scipy.io as scio
import scipy.io
def pointwiseMissing(dense_tensor,missingrate):
    dim1,dim2,dim3=dense_tensor.shape
    sparse_tensor = dense_tensor.copy()
    missingMatrix = np.random.uniform(0, 1, (dim1,dim3))
    binary_mat = np.round(missingMatrix + 0.5 - missingrate)
    for i in range(dim2):
        sparse_tensor[:, i, :] = dense_tensor[:, i, :] * binary_mat
    np.save("pems08_pm_{}.npy" .format(missingrate), sparse_tensor)
    return sparse_tensor
def continuousMissing(dense_tensor,missingrate):
    dim1, dim2, dim3 = dense_tensor.shape
    sparse_tensor = dense_tensor.copy()
    missingMatrix = np.random.uniform(0, 1, (dim1,dim3))
    binary_mat = np.round(missingMatrix + 0.5 - missingrate)
    dense_tensor_reshape = dense_tensor.reshape(dim1, dim2,-1, 288)
    sparse_tensor_reshape = dense_tensor_reshape.copy()
    for i in range(dim1):
        for j in range(158):
            #print(len(sparse_tensor_reshape[0][0][0]),'maxdim')
            sparse_tensor_reshape[i, :, j, :] = dense_tensor_reshape[i, :, j, :] * binary_mat[i, j]
    sparse_tensor = sparse_tensor_reshape.reshape(dim1, dim2, dim3)
    np.save("CD_cm_{}.npy" .format(missingrate), sparse_tensor)
    return sparse_tensor
data = scipy.io.loadmat('../datasets/Chengdu/Chengdu.mat')['tensor']
data_p=data.reshape(826,1,-1)
for i in range(1,10):
    print(data_p.shape)
    #data_q=np.load("pems08_cm_10.npy")
    #print(data_q.shape)
    missingrate=i/10
    print(missingrate)
    #pointwiseMissing(data_p,missingrate)
    continuousMissing(data_p,missingrate)
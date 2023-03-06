import numpy as np
def pointwiseMissing(dense_tensor,missingrate):
    dim1,dim2,dim3=dense_tensor.shape
    sparse_tensor = dense_tensor.copy()
    missingMatrix = np.random.uniform(0, 1, (dim1,dim3))
    binary_mat = np.round(missingMatrix + 0.5 - missingrate)
    for i in range(dim2):
        sparse_tensor[:, i, :] = dense_tensor[:, i, :] * binary_mat
    np.save("pems04_pm_random_mat_%.2f%%.npy" % missingrate, sparse_tensor)
    return sparse_tensor
def continuousMissing(dense_tensor,missingrate):
    dim1, dim2, dim3 = dense_tensor.shape
    sparse_tensor = dense_tensor.copy()
    missingMatrix = np.random.uniform(0, 1, (dim1,dim3))
    binary_mat = np.round(missingMatrix + 0.5 - missingrate)
    dense_tensor_reshape = dense_tensor.reshape(dim1, dim2,-1, 144)
    sparse_tensor_reshape = dense_tensor_reshape.copy()
    for i in range(dim1):
        for j in range(62):
            sparse_tensor_reshape[i, :, j, :] = dense_tensor_reshape[i, :, j, :] * binary_mat[i, j]
    sparse_tensor = sparse_tensor_reshape.reshape(dim1, dim2, dim3)
    np.save("pems04_cm_random_mat_%.2f%%.npy" % missingrate, sparse_tensor)
    return sparse_tensor
data=np.load("../datasets/PEMS08/pems04.npz")
data_p=data['data'].transpose(1,2,0)
print(data_p.shape)
missingrate=0.4
pointwiseMissing(data_p,missingrate)
continuousMissing(data_p,missingrate)
import numpy as np
import csv
directory = '../datasets/PEMS04/'
dense_tensor = np.load( directory + 'pems04.npz')
dense_tensor=dense_tensor['data'].transpose(1,2,0)
dim1,dim2,dim3=dense_tensor.shape
A=np.zeros((dim1,dim1))
W=np.zeros((dim1,dim1))
firstMark=0
with open(directory+'distance.csv') as f1:
    reader=csv.reader(f1)
    for raw in reader:
        if firstMark==0:
            firstMark=1
            continue
        A[int(raw[0])][int(raw[1])]=1
        W[int(raw[0])][int(raw[1])]=float(raw[2])
np.save('../datasets/PEMS04/A04.npy', A)
np.save('../datasets/PEMS04/D04.npy', W)

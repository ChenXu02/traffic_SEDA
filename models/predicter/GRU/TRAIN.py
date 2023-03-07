import torch
import torch.nn as nn
import math
import numpy as np
from OP import Optim
from sklearn.metrics import mean_squared_error,mean_absolute_error
from GRU import Model
from DATA import Data_utility
import time
import os
import joblib
from metrics import All_Metrics

data = Data_utility(file_name1='pems08.npz', train=0.7, valid=0.15, horizon=1, window=12, normalize=1)#GRU数据
print("data process done")
m=data.mean
s=data.std
max=data.max
trainstart=data.trainstartnum
vstart=data.validstartnum#对 get_traffic有用
tstart=data.teststartnum

batch_size=64
windows=data.train[0].shape[1]#GRU部分参数
n_val=data.train[0].shape[2]#GRU部分参数y_true[pos][0] , y_pred[pos][0]
hidRNN=64
epochs = 100
best_val=10000000
def mean_absolute_percentage_error(y_true, y_pred, pos):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true[pos] - y_pred[pos]) / y_true[pos])) * 100
def Regularization(model):
    L2=0
    for param in model.parameters():
        L2+=torch.norm(param,2)
    return L2
def get_batches(inputs,inputs2,batch_size):
    data=[]
    label=[]
    length = len(inputs)
    start_idx = 0
    while (start_idx+batch_size < length):
        end_idx = min(length, start_idx + batch_size)
        data.append(inputs[start_idx:end_idx])
        label.append(inputs2[start_idx:end_idx])
        start_idx += batch_size
    return data,label
def train(data, batch_size, model, criterion, optim):
    model.train()
    total_loss = 0
    n_samples = 0
    trabatch,tralabelbatch=get_batches(data.train[0],data.train[1], batch_size)
    for i in range(0,len(trabatch)):#tra为一个batch
        t1 = time.time()
        print("****************第   ",i,"   个batch***************")
        tralabel=tralabelbatch[i].clone().detach()
        model.zero_grad()
        tralabelloss=tralabel
        output = model(trabatch[i])
        loss = criterion(output, tralabelloss)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.item()
        n_samples+=1
        print(time.time() - t1, "time")
        print("loss:",loss.item())
    return total_loss /n_samples
def evaluate(data,s,m,max, model, batch_size):
    model.eval()
    num=0
    n_samples = 0
    predict = None
    test = None
    trabatch,tralabelbatch=get_batches(data[0],data[1],batch_size)
    for i in range(0,len(trabatch)):
        tralabel = tralabelbatch[i].clone().detach()
        ###model###
        with torch.no_grad():
            output= model(trabatch[i])
        if predict is None:
            predict = output
            test = tralabel
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, tralabel))
        n_samples += (output.size(0))
        num+=1
    #predict = (predict.data.cpu().numpy()*s)+m
    #Ytest = (test.data.cpu().numpy()*s)+m
    predict = (predict.data.cpu().numpy() * max)
    Ytest = (test.data.cpu().numpy() * max)
    mae, rmse, mape, _, _ = All_Metrics(predict, Ytest,None, 0)
    #pos = np.where(Ytest != 0)
    #rmse = math.sqrt(mean_squared_error(Ytest, predict))
    #mae = mean_absolute_error(Ytest , predict)
    #mape = mean_absolute_percentage_error(Ytest, predict, pos)
    return  predict,rmse, mae, mape

if (torch.cuda.is_available()):
    device = torch.device("cuda:1")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
model = Model(n_val,windows,hidRNN)
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)#7366:TGCN       this 7453
criterion = nn.MSELoss()#(y-x)**2
criterion = criterion.cuda()
optimizer = Optim(
    model.parameters(), 'adam', lr=0.001, max_grad_norm=10, start_decay_at=0, lr_decay=0.8
)
print('begin training')
dirs = 'testModel'
if not os.path.exists(dirs):
    os.makedirs(dirs)
for epoch in range(0, epochs):
    epoch_start_time = time.time()
    train_loss = train(data, batch_size, model, criterion, optimizer)
    predict,rmse, mae,  mape = evaluate(data.valid ,s,m,max, model, batch_size)
    print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | lr {:5.4f} |  rmse {:5.4f} | mae {:5.4f} |   mape  {:5.4f} |'
        .format(epoch, (time.time() - epoch_start_time), train_loss, optimizer.lr,rmse, mae,  mape))
    if rmse < best_val:
        best_val = rmse
    optimizer.updateLearningRate(rmse, epoch)
    if epoch%5==0:
        predictT, rmseT, maeT, mapeT = evaluate(data.test,s,m, max, model,batch_size)
        print('|  Trmse {:5.4f} | Tmae {:5.4f} |   mapeT  {:5.4f} |'.format( rmseT, maeT,  mapeT))
    joblib.dump(model, dirs + '/LR.pkl')
predictT, rmseT, maeT,  mapeT = evaluate(data.test,s,m, max, model,  batch_size)
print( '|  Trmse {:5.4f} | Tmae {:5.4f} |   mapeT  {:5.4f} |' .format(  rmseT, maeT,  mapeT))



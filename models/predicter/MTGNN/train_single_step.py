import argparse
import math
import time

import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib

from util import *
from trainer import Optim
from metrics import All_Metrics
import csv
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size,mark):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    iter=0
    for X, Y in data.get_batches(X, Y, batch_size, False):
        print(iter)
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape)==1:
            output = output.unsqueeze(dim=0)
        scale = data.scale.expand(output.size(0), data.m)
        scale_o = data.scale_o.expand(output.size(0), data.m)
        if predict is None:
            predict = output* scale
            test = Y*scale_o
        else:
            predict = torch.cat((predict, output* scale))
            test = torch.cat((test, Y*scale_o))
        total_loss += evaluateL2(output * scale, Y * scale_o).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale_o).item()
        n_samples += (output.size(0) * data.m)
        iter+=1

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    mae, rmse, mape, _, _ = All_Metrics(predict, Ytest, None, 0)
    if mark==2:
        with open('pems08_MTGNN_result_0314.csv', 'a+', newline='') as f1:
            writer = csv.writer(f1)
            writer.writerow(
                ["filename:", args.filename, "RMSE", rmse, "MAE", mae, "MAPE", mape,
                 'prediction'])

    return rmse, mae, mape


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        if iter%10==0:
            print(iter,'iter')
            print(X.size())
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx,id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:,id]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter%100==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples



def main():

    Data = DataLoaderS(args.data,args.data_o, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.normalize)

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False)
    model = model.to(device)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)


    best_val = 10000000
    optim = Optim(
        model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            print('start to evaluate')
            val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                               args.batch_size,1)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            if val_loss < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
            if epoch % 1 == 0:
                test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                                     args.batch_size,2)
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr), flush=True)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        print('load:',args.save)
    test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size,2)
    print("final test rmse {:5.4f} | test mae {:5.4f} | test mape {:5.4f}".format(test_acc, test_rae, test_corr))
    return  test_acc, test_rae, test_corr

if __name__ == "__main__":

    for j in {'cm', 'pm'}:
        for i in range(0, 10):
            if i == 0:
                filename = 'pems08_{}_0.npy'.format(j)
            else:
                filename = 'pems08_{}_0.{}.npy'.format(j, i)
            print(filename)
            parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
            parser.add_argument('--data', type=str, default='../LSTM-GL-ReMF/'+filename,
                                help='location of the data file')
            parser.add_argument('--data_o', type=str, default='../LSTM-GL-ReMF/pems08_pm_0.npy',
                                help='location of the data file')
            parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                                help='report interval')
            parser.add_argument('--save', type=str, default='model/model'+filename+'.pt',
                                help='path to save the final model')
            parser.add_argument('--optim', type=str, default='adam')
            parser.add_argument('--filename', type=str, default=filename)
            parser.add_argument('--L1Loss', type=bool, default=True)
            parser.add_argument('--normalize', type=int, default=2)
            parser.add_argument('--device', type=str, default='cpu', help='')
            parser.add_argument('--gcn_true', type=bool, default=True,
                                help='whether to add graph convolution layer')
            parser.add_argument('--buildA_true', type=bool, default=True,
                                help='whether to construct adaptive adjacency matrix')
            parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
            parser.add_argument('--num_nodes', type=int, default=170, help='number of nodes/variables')
            parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
            parser.add_argument('--subgraph_size', type=int, default=20, help='k')
            parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
            parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
            parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
            parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
            parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
            parser.add_argument('--end_channels', type=int, default=64, help='end channels')
            parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
            parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
            parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')
            parser.add_argument('--horizon', type=int, default=1)
            parser.add_argument('--layers', type=int, default=5, help='number of layers')

            parser.add_argument('--batch_size', type=int, default=64, help='batch size')
            parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
            parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

            parser.add_argument('--clip', type=int, default=5, help='clip')

            parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
            parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

            parser.add_argument('--epochs', type=int, default=10, help='')
            parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
            parser.add_argument('--step_size', type=int, default=100, help='step_size')

            args = parser.parse_args()
            device = torch.device(args.device)
            torch.set_num_threads(3)
            test_acc, test_rae, test_corr = main()

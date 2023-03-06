import numpy as np
import scipy.io
from numpy.linalg import inv as inv
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
import time
def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)
def cp_combine(U, V, X):
    return np.einsum('is, js, ts -> ijt', U, V, X)
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
def graph_weight_cal(dist, epsilon = 0.5):
    dim = dist.shape[0]
    distances = dist[np.nonzero(dist)].flatten()
    std = distances.std()
    print(std)
    std_square = 5 * std ** 2
    A = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if dist[i][j] > 0:
                weight = np.exp(- dist[i][j] ** 2 / std_square)
                if i != j and weight >= epsilon:
                    A[i][j] = weight
    return A
def create_lstm_samples(dataset, time_lags, rate):
    dataX, dataY = [], []
    data_len = dataset.shape[0] - np.max(time_lags)
    t_sample = np.random.choice(data_len, int(rate * data_len), replace=False)

    for t in t_sample:
        a = dataset[t + np.max(time_lags) - time_lags, :][::-1]
        dataX.append(a)
        dataY.append(dataset[t + np.max(time_lags), :])
    return np.array(dataX), np.array(dataY)
def lstmmodel(rank, lag_len):
    # create the LSTM network
    model = Sequential()
#     model.add(LSTM(rank, input_shape = (lag_len, rank), return_sequences = True)) # If you need multi-layer LSTM
    model.add(LSTM(rank, input_shape = (lag_len, rank)))
    model.add(Dense(rank))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def mean_absolute_percentage_error(y_true, y_pred, pos):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true[pos] - y_pred[pos]) / y_true[pos])) * 100
def root_mean_squared_error(y_true, y_pred, pos):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square(y_true[pos] - y_pred[pos])))
def LSTM_GL_ReTF(sparse_tensor, invD, init, time_lags, lambda_w, lambda_g, lambda_v, lambda_x, eta, sampling_rate,
                 maxiter, track, dense_tensor=0):
    W = init["W"]
    V = init["V"]
    X = init["X"]
    dim1, dim2, dim3 = sparse_tensor.shape
    binary_tensor = np.zeros((dim1, dim2, dim3))
    position = np.where((sparse_tensor != 0))
    binary_tensor[position] = 1
    d = len(time_lags)
    max_lags = np.max(time_lags)
    r = X.shape[1]
    if track:
        pos = np.where((sparse_tensor == 0) & (dense_tensor != 0))
    model = lstmmodel(r, d)
    model_reverse = lstmmodel(r, d)
    start_time = time.time()

    for iters in range(maxiter):
        var1 = kr_prod(X, V).T  # (r, n*t)
        var2 = kr_prod(var1, var1)  # (r*r, n*t)
        mat1 = ten2mat(binary_tensor, 0).T  # (n*t, m)
        mat2 = ten2mat(sparse_tensor, 0).T  # (n*t, m)
        for i in range(dim1):
            vec1 = np.matmul(W.T, invD[i, :])
            var_Lambda1 = np.matmul(var2, mat1[:, i]).reshape([rank, rank]) + lambda_w * np.eye(
                rank) + lambda_g * np.eye(rank) * np.sum(invD[i, :])
            inv_var_Lambda1 = np.linalg.inv((var_Lambda1 + var_Lambda1.T) / 2)
            W[i, :] = np.matmul(inv_var_Lambda1, np.matmul(var1, mat2[:, i]) + lambda_g * vec1)

        var1 = kr_prod(X, W).T
        var2 = kr_prod(var1, var1)
        mat1 = ten2mat(binary_tensor, 1).T  # (m*t, n)
        mat2 = ten2mat(sparse_tensor, 1).T  # (m*t, n)
        for j in range(dim2):
            var_Lambda1 = np.matmul(var2, mat1[:, j]).reshape([rank, rank]) + lambda_v * eta * np.eye(rank)
            inv_var_Lambda1 = np.linalg.inv((var_Lambda1 + var_Lambda1.T) / 2)
            V[j, :] = np.matmul(inv_var_Lambda1, np.matmul(var1, mat2[:, j]))

        var1 = kr_prod(V, W).T  # (r, m*n)
        var2 = kr_prod(var1, var1)  # (r*r, m*n)
        mat1 = ten2mat(binary_tensor, 2).T  # (m*n, n)
        mat2 = ten2mat(sparse_tensor, 2).T  # (m*n, n)
        for t in range(dim3):
            if iters == 0 or t < max_lags:
                var_Lambda1 = np.matmul(var2, mat1[:, t]).reshape([rank, rank]) + lambda_x * eta * np.eye(rank)
                X[t, :] = np.matmul(inv((var_Lambda1 + var_Lambda1.T) / 2), np.matmul(var1, mat2[:, t]))
            else:
                var_Lambda1 = np.matmul(var2, mat1[:, t]).reshape([rank, rank]) + lambda_x * np.eye(
                    rank) + lambda_x * eta * np.eye(rank)
                X_hat = X[t - time_lags, :][::-1]
                X_hat_feed = X_hat[np.newaxis, :, :]
                Qt = model.predict(X_hat_feed,verbose=0)[0]
                X[t, :] = np.matmul(inv((var_Lambda1 + var_Lambda1.T) / 2),
                                    (np.matmul(var1, mat2[:, t]) + lambda_x * Qt))

        if iters == 0:
            lstmX, lstmY = create_lstm_samples(X, time_lags, 1)
            model.fit(lstmX, lstmY, epochs=20, batch_size=50, verbose=0)
        else:
            lstmX, lstmY = create_lstm_samples(X, time_lags, sampling_rate)
            model.fit(lstmX, lstmY, epochs=2, batch_size=10, verbose=0)
        if (iters + 1) % 1 == 0:
            print('Iterations: %d, time cost: %ds' % ((iters + 1), (time.time() - start_time)))
            start_time = time.time()
            if track:
                tensor_hat = cp_combine(W, V, X)
                tensor_hat[position] = sparse_tensor[position]
                tensor_hat[tensor_hat < 0] = 0
                rmse = root_mean_squared_error(dense_tensor, tensor_hat, pos)
                mape = mean_absolute_percentage_error(dense_tensor, tensor_hat, pos)
                print(np.mean(W))
                print(np.mean(V))
                print(np.mean(X))
                print('Imputation RMSE = %.2f' % rmse)
                print('Imputation MAPE = %.2f' % mape)
            print()
    #     model.save('model_save\lstm_trained1.h5')
    tensor_hat = cp_combine(W, V, X)
    tensor_hat[position] = sparse_tensor[position]
    tensor_hat[tensor_hat < 0] = 0
    return tensor_hat, W, V, X, model
def OnlineLSTMReTF(sparse_mat, init, lambda_x, time_lags):
    time_lags = time_lags[::-1]
    W = init["W"]
    V = init["V"]
    X = init["X"]
    model = init["model"]
    dim1, dim2 = sparse_mat.shape
    t, rank = X.shape
    X_hat = X[t - 1 - time_lags, :].copy()
    X_hat_feed = X_hat[np.newaxis, :, :]
    Qt = model.predict(X_hat_feed,verbose=0)[0]

    sparse_tensor = np.zeros((dim1, dim2, 1))
    sparse_tensor[:, :, 0] = sparse_mat
    position = np.where(sparse_tensor != 0)
    binary_tensor = np.zeros(sparse_tensor.shape)
    binary_tensor[position] = 1
    var1 = kr_prod(V, W).T
    var2 = kr_prod(var1, var1)
    var_mu = np.matmul(var1, ten2mat(sparse_tensor, 2).reshape([dim1 * dim2])) + lambda_x * Qt
    inv_var_Lambda = inv(
        np.matmul(var2, ten2mat(binary_tensor, 2).reshape([dim1 * dim2])).reshape([rank, rank]) + lambda_x * np.eye(
            rank))
    return np.matmul(inv_var_Lambda, var_mu)
def online_prediction(sparse_tensor, init, time_lags, lambda_x, maxiter):
    W = init["W"]
    V = init["V"]
    X = init["X"]
    model = init["model"]
    pre_step_num = X.shape[0]
    rank = X.shape[1]
    dim1, dim2, dim3 = sparse_tensor.shape
    X_hat = np.zeros((dim3 + pre_step_num, rank))
    tensor_pred = np.zeros((dim1, dim2, dim3))
    X_hat[:pre_step_num, :] = X.copy()
    start_time = time.time()
    for t in range(dim3):
        if t == 0:
            X_star = X_hat[pre_step_num + t - time_lags, :][::-1]
            X_star_feed = X_star[np.newaxis, :, :]
            Qt = model.predict(X_star_feed,verbose=0)[0]
            X_hat[pre_step_num + t, :] = Qt.copy()
        else:
            sparse_mat = sparse_tensor[:, :, t - 1]
            if np.where(sparse_mat > 0)[0].shape[0] > 0:
                init = {"W": W, "V": V, "X": X_hat[pre_step_num + t - np.max(time_lags) - 1: pre_step_num + t, :],
                        "model": model}
                X_c = OnlineLSTMReTF(sparse_mat, init, lambda_x / dim3, time_lags)
                X_hat[pre_step_num + t - 1, :] = X_c.copy()
                X_star = X_hat[pre_step_num + t - time_lags, :][::-1]
                X_star_feed = X_star[np.newaxis, :, :]
                Qt = model.predict(X_star_feed,verbose=0)[0]
                X_hat[pre_step_num + t, :] = Qt.copy()
            else:
                X_star = X_hat[pre_step_num + t - time_lags, :][::-1]
                X_star_feed = X_star[np.newaxis, :, :]
                Qt = model.predict(X_star_feed,verbose=0)[0]
                X_hat[pre_step_num + t, :] = Qt.copy()
        tensor_pred[:, :, t] = np.einsum('ir, jr, r -> ij', W, V, X_hat[pre_step_num + t, :])
        if (t + 1) % 1000 == 0:
            print('Time step: %d, time cost: %d s' % ((t + 1), (time.time() - start_time)))
            start_time = time.time()

    sparse_mat = sparse_tensor[:, :, -1]
    init = {"W": W, "V": V, "X": X_hat[dim2 + pre_step_num - np.max(time_lags) - 1:, :], "model": model}
    X_c = OnlineLSTMReTF(sparse_mat, init, lambda_x / dim2, time_lags)
    X_hat[dim2 + pre_step_num - 1, :] = X_c.copy()
    tensor_rec = cp_combine(W, V, X_hat[pre_step_num:, :])
    return tensor_rec, tensor_pred



directory = '../datasets/PEMS08/'

missing_rate = 0.1
mode = 'CM'

Dist = np.load(directory + 'Dist.npy')
A = np.load(directory + 'Adj.npy')
dense_tensor = np.load( directory + 'PeMS08.npy')

dim1, dim2, dim3 = dense_tensor.shape
# dense_mat = dense_tensor.reshape(dim1 * dim2, dim3)
# =============================================================================
### Point-wise missing (PM) scenario
### Set the PM scenario by:
if mode == 'PM':
    pm_random_mat = np.load(directory + 'pm_random_mat.npy')
    binary_mat = np.round(pm_random_mat + 0.5 - missing_rate)
    sparse_tensor = dense_tensor.copy()
    for i in range(dim2):
        sparse_tensor[:, i, :] = dense_tensor[:, i, :] * binary_mat
# =============================================================================
# =============================================================================
### Continuous-random missing (CM) scenario
### Set the CM scenario by:
if mode == 'CM':
    cm_random_mat = np.load(directory + 'cm_random_mat.npy')
    binary_mat = np.round(cm_random_mat + 0.5 - missing_rate)
    dense_tensor_reshape = dense_tensor.reshape(dim1, dim2, 62, 288)
    sparse_tensor_reshape = dense_tensor_reshape.copy()
    for i in range(dim1):
        for j in range(62):
            sparse_tensor_reshape[i, :, j, :] = dense_tensor_reshape[i, :, j, :] * binary_mat[i, j]
    sparse_tensor = sparse_tensor_reshape.reshape(dim1, dim2, dim3)
# =============================================================================
print('Missing rate = %s %.1f'%(mode, missing_rate))


sparse_tensor = np.expand_dims(sparse_tensor[:, 0, 8928:], axis=1)
dense_tensor = np.expand_dims(dense_tensor[:, 0, 8928:], axis=1)
print(sparse_tensor.shape,'shape')
dim1, dim2, dim3 = sparse_tensor.shape
#Creating training set and test set
test_len = 2880
train_len = dim3 - test_len
training_set = sparse_tensor[:, :, :train_len]
test_set = sparse_tensor[:, :, train_len:]
print('The size of training set is:')
print(training_set.shape)
print()
print('The size of test set is:')
print(test_set.shape)
#Saving ground truth(real value)
training_ground_truth = dense_tensor[:, :, :train_len]
test_ground_truth = dense_tensor[:, :, train_len:]
print('The size of training set ground truth is:')
print(training_ground_truth.shape)
print()
print('The size of test set ground truth is:')
print(test_ground_truth.shape)


np.random.seed(seed=10)
rank = 20
maxiter = 100
eta = 0.2
lambda_w = 100
lambda_g = 100
lambda_v = 100
lambda_x = 100
sampling_rate = 1.0
time_lags = np.array([1, 2, 288])#np.arange(1, 25, 1)
track = True
dim1, dim2, dim3 = training_set.shape
init = {"W": 0.1 * np.random.rand(dim1, rank), "V": 0.1 * np.random.rand(dim2, rank), "X": 0.1 * np.random.rand(dim3, rank)}
tensor_hat, W, V, X, model = LSTM_GL_ReTF(training_set, A, init, time_lags, lambda_w, lambda_g, lambda_v,
                                       lambda_x, eta, sampling_rate, maxiter, track, training_ground_truth)


start_time = time.time()
sc = 100000
init = {"W": W, "V": V, "X": X[- np.max(time_lags): , :], "model": model}
test_mat_rec, test_mat_pred = online_prediction(test_set, init, time_lags, sc * lambda_x, maxiter)
print('Shape of imputed data is:')
print(test_mat_rec.shape)
print()
print('Shape of predicted data is:')
print(test_mat_pred.shape)

print('10 first prediciton on test set:')
print(test_mat_pred[0, 0, :10])
print()
print('10 first real value on test set')
print(test_ground_truth[0, 0, :10])

# Prediction
pos = np.where(test_ground_truth != 0)
testPred_rmse = root_mean_squared_error(test_ground_truth, test_mat_pred, pos)
print('Test prediction RMSE: %.2f RMSE' % (testPred_rmse))
testPred_mape = mean_absolute_percentage_error(test_ground_truth, test_mat_pred, pos)
print('Test prediction MAPE: %.2f%% MAPE' % (testPred_mape))
print()


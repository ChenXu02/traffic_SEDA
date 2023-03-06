import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from numpy.linalg import inv as inv
from numpy.linalg import solve as solve
from numpy.linalg import cholesky as cholesky_lower
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut
import scipy.io as scio
def mvnrnd_pre(mu, Lambda):
    src = normrnd(size = (mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False),
                    src, lower = False, check_finite = False, overwrite_b = True) + mu
def cp_combine(var):
    return np.einsum('is, js, ts -> ijt', var[0], var[1], var[2])
## 1st solution
def vec_combine(vector):
    tensor = 0
    d = len(vector)
    for i in range(d):
        ax = [len(vector[i]) if j == i else 1 for j in range(d)]
        tensor = tensor + vector[i].reshape(ax, order = 'F')
    return tensor
## 2nd solution
def vec_combine(vector):
    return (vector[0][:, np.newaxis, np.newaxis] + vector[1][np.newaxis, :, np.newaxis]
            + vector[2][np.newaxis, np.newaxis, :])
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat
def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]
def compute_rmse(var, var_hat):
    return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])
def sample_global_mu(mu_sparse, pos_obs, tau_eps, tau0 = 1):
    tau_tilde = 1 / (tau_eps * len(pos_obs[0]) + tau0)
    mu_tilde = tau_eps * np.sum(mu_sparse) * tau_tilde
    return np.random.normal(mu_tilde, np.sqrt(tau_tilde))
def sample_bias_vector(bias_sparse, factor, bias, ind, dim, k, tau_eps, tau0 = 1):
    for k in range(len(dim)):
        idx = tuple(filter(lambda x: x != k, range(len(dim))))
        temp = vector.copy()
        temp[k] = np.zeros((dim[k]))
        tau_tilde = 1 / (tau_eps * bias[k] + tau0)
        mu_tilde = tau_eps * np.sum(ind * (bias_sparse - vec_combine(temp)), axis = idx) * tau_tilde
        vector[k] = np.random.normal(mu_tilde, np.sqrt(tau_tilde))
    return vector
def sample_factor(tau_sparse, factor, ind, dim, k, tau_eps, beta0=1):
    dim, rank = factor[k].shape
    dim = factor[k].shape[0]
    factor_bar = np.mean(factor[k], axis=0)
    temp = dim / (dim + beta0)
    var_mu_hyper = temp * factor_bar
    var_W_hyper = inv(np.eye(rank) + cov_mat(factor[k], factor_bar) + temp * beta0 * np.outer(factor_bar, factor_bar))
    var_Lambda_hyper = wishart.rvs(df=dim + rank, scale=var_W_hyper)
    var_mu_hyper = mvnrnd_pre(var_mu_hyper, (dim + beta0) * var_Lambda_hyper)

    idx = list(filter(lambda x: x != k, range(len(factor))))
    var1 = kr_prod(factor[idx[1]], factor[idx[0]]).T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ ten2mat(tau_eps * ind, k).T).reshape([rank, rank, dim]) + var_Lambda_hyper[:, :, np.newaxis]
    var4 = var1 @ ten2mat(tau_sparse, k).T + (var_Lambda_hyper @ var_mu_hyper)[:, np.newaxis]
    for i in range(dim):
        factor[k][i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    return factor[k]
def sample_precision_tau(error_tensor, pos_obs):
    var_alpha = 1e-6 + 0.5 * len(pos_obs[0])
    var_beta = 1e-6 + 0.5 * np.linalg.norm(error_tensor, 2) ** 2
    return np.random.gamma(var_alpha, 1 / var_beta)
def BATF_Gibbs(dense_tensor, sparse_tensor, vector, factor, burn_iter, gibbs_iter):
    """Bayesian Augmented Tensor Factorization (BATF) with Gibbs sampling."""

    dim = np.array(sparse_tensor.shape)
    rank = factor[0].shape[1]
    if np.isnan(sparse_tensor).any() == False:
        ind = sparse_tensor != 0
        pos_obs = np.where(ind)
        pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    elif np.isnan(sparse_tensor).any() == True:
        pos_test = np.where((dense_tensor != 0) & (np.isnan(sparse_tensor)))
        ind = ~np.isnan(sparse_tensor)
        pos_obs = np.where(ind)
        sparse_tensor[np.isnan(sparse_tensor)] = 0
    num_obs = len(pos_obs[0])
    dense_test = dense_tensor[pos_test]
    scio.savemat('sparse_tensor_missing_10%.mat', {'tensor': sparse_tensor})
    del dense_tensor

    show_iter = 200
    tau_eps = 1
    bias = []
    for k in range(len(dim)):
        idx = tuple(filter(lambda x: x != k, range(len(dim))))
        bias.append(np.sum(ind, axis = idx))
    temp = cp_combine(factor)
    temp_hat = np.zeros(len(pos_test[0]))
    tensor_hat_plus = np.zeros(dim)
    for it in range(burn_iter + gibbs_iter):
        print(it)
        temp = sparse_tensor - temp
        mu_glb = sample_global_mu(temp[pos_obs] - vec_combine(vector)[pos_obs], pos_obs, tau_eps)
        vector = sample_bias_vector(temp - mu_glb, factor, bias, ind, dim, k, tau_eps)
        del temp
        tau_sparse = tau_eps * ind * (sparse_tensor - mu_glb - vec_combine(vector))
        for k in range(len(dim)):
            factor[k] = sample_factor(tau_sparse, factor, ind, dim, k, tau_eps)
        temp = cp_combine(factor)
        tensor_hat = mu_glb + vec_combine(vector) + temp
        if it==1000:
            scio.savemat('sparse_tensor_imputation_10%.mat', {'tensor': temp_hat})
        temp_hat += tensor_hat[pos_test]
        #print(sparse_tensor,'sparse')

        tau_eps = sample_precision_tau(sparse_tensor[pos_obs] - tensor_hat[pos_obs], pos_obs)
        if it + 1 > burn_iter:
            tensor_hat_plus += tensor_hat
        if (it + 1) % show_iter == 0 and it < burn_iter:
            temp_hat = temp_hat / show_iter
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(compute_mape(dense_test, temp_hat)))
            print('RMSE: {:.6}'.format(compute_rmse(dense_test, temp_hat)))
            temp_hat = np.zeros(len(pos_test[0]))
            print()
    tensor_hat = tensor_hat_plus / gibbs_iter
    print('Imputation MAPE: {:.6}'.format(compute_mape(dense_test, tensor_hat[pos_test])))
    print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_test, tensor_hat[pos_test])))
    print()

    return tensor_hat, mu_glb, vector, factor

import time
import time
import scipy.io
import numpy as np
np.random.seed(1000)
print('start')
dense_tensor = scipy.io.loadmat('datasets/Chengdu/Chengdu.mat')['tensor']
dim = dense_tensor.shape
missing_rate = 0.1 # Random missing (RM)
sparse_tensor = dense_tensor * np.round(np.random.rand(dim[0], dim[1], dim[2]) + 0.5 - missing_rate)

start = time.time()
dim = np.array(sparse_tensor.shape)
rank = 80
vector = []
factor = []
for k in range(len(dim)):
    vector.append(0.1 * np.random.randn(dim[k],))
    factor.append(0.1 * np.random.randn(dim[k], rank))
burn_iter = 1000
gibbs_iter = 200
BATF_Gibbs(dense_tensor, sparse_tensor, vector, factor, burn_iter, gibbs_iter)
end = time.time()
print('Running time: %d seconds'%(end - start))

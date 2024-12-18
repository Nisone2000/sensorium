import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import h5py
import scipy.io

import scipy
import scipy.stats as stats

dtype = tf.float64
dtype_convert = tf.to_double

import time


def normalise_data(X, shouldNormalise):
    if shouldNormalise:
        X_mean = np.mean(X, axis=0)
        X = X - X_mean
        X_std = X.std(axis=0, ddof=1, keepdims=True)
        X_std[X_std < 1e-10] = 1
        X_std = np.maximum(X_std, 1e-8)
        X = X / X_std
        assert (np.all(X != np.NaN))
    else:
        X_mean = 0
        X_std = 1

    return X, X_mean, X_std


def standard_pca_initialisation(Y, Q):
    assert (Y.shape[0] >= Q)
    assert (Y.shape[1] >= Q)

    Y_norm, Y_mean, Y_std = normalise_data(Y, shouldNormalise=True)

    if Y.shape[0] < (Q + 2):
        W, V = scipy.linalg.eig(np.dot(Y_norm, Y_norm.T))
        V = V[:, 0:Q]
    else:
        W, V = scipy.sparse.linalg.eigs(np.dot(Y_norm, Y_norm.T), Q)

    X0 = np.real(V)
    X0 = X0 / np.mean(X0.std(axis=0, ddof=1))

    return X0


def log_det_from_chol(L):
    return 2.0 * tf.reduce_sum(tf.log(tf.diag_part(L)))


def real_variable(inital_value, prior_function=None, name=None):
    t_a = tf.Variable(inital_value, dtype=dtype, name=name)
    return t_a


def positive_variable(inital_value, prior_function=None, name=None):
    t_a = tf.Variable(tf.log(tf.cast(inital_value, dtype=dtype)), dtype=dtype, name=name)
    return tf.exp(t_a)


# Will ensure a monotonic sequence in the range [-1,1] without the end points hard set to -1 and 1.
# def monotonic_sequence_variable(N_samples, init, min=-1.0, max=1.0):
#     assert issubclass(type(init), np.ndarray)
#     s = list(np.shape(init))
#     s[0] = 1
#     init = np.concatenate([-10.0 * np.ones(s), init])
#     t_raw = real_variable(init)
#     t_sm = tf.nn.softmax(t_raw, dim=0)
#     t_seq = (max - min) * tf.cumsum(t_sm, axis=0, exclusive=True) + min
#     t_seq = t_seq[1:, ]
#     return t_seq, t_raw

def monotonic_sequence_variable(N_samples, init, min=-1.0, max=1.0):
    assert issubclass(type(init), np.ndarray)
    init = np.random.randn(N_samples - 1, 1)
    t_raw = real_variable(init)
    t_sm = tf.nn.softmax(t_raw, axis=0)
    t_seq = (max - min) * tf.cumsum(t_sm, axis=0, exclusive=True) + min
    t_seq = tf.concat([t_seq, [[1]]], axis=0)
    return t_seq, t_raw

class Kernel:
    def __init__(self, covar_matrix_func, covar_diag_func, descriptor, kernels=None):
        self._descriptor = descriptor
        self._covar_matrix_func = covar_matrix_func
        self._covar_diag_func = covar_diag_func
        self._kernels = kernels

    @property
    def descriptor(self):
        return self._descriptor

    def covar_matrix(self, t_X, t_Z):
        return self._covar_matrix_func(t_X=t_X, t_Z=t_Z)

    def covar_diag(self, t_X):
        return self._covar_diag_func(t_X=t_X)


def create_sum_kernel(kernels): # e.g. kernels=[create_squared_exp_kernel(..), create_matern32_kernel(..)]
    assert all([issubclass(type(k), Kernel) for k in kernels])

    def matrix_func(t_X, t_Z):
        return tf.reduce_sum(tf.stack([k.covar_matrix(t_X, t_Z) for k in kernels]), axis=0)

    def diag_func(t_X):
        return tf.reduce_sum(tf.stack([k.covar_diag(t_X) for k in kernels]), axis=0)

    descriptor = 'Sum (' + ', '.join([k.descriptor for k in kernels]) + ')'
    return Kernel(descriptor=descriptor,
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=kernels)


def create_prod_kernel(kernels): # e.g. kernels=[create_squared_exp_kernel(..), create_matern32_kernel(..)]
    assert all([issubclass(type(k), Kernel) for k in kernels])

    def matrix_func(t_X, t_Z):
        return tf.reduce_prod(tf.stack([k.covar_matrix(t_X, t_Z) for k in kernels]), axis=0)

    def diag_func(t_X):
        return tf.reduce_prod(tf.stack([k.covar_diag(t_X) for k in kernels]), axis=0)

    descriptor = 'Product (' + ', '.join([k.descriptor for k in kernels]) + ')'
    return Kernel(descriptor=descriptor,
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=kernels)

def create_squared_exp_kernel(t_alpha, t_gamma):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        sq_dist_xz = xx + tf.transpose(zz) - t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        return t_alpha * tf.exp(- sq_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Squared Exponential',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)

def create_periodic_kernel(t_alpha, t_gamma, t_period): # inverse period
    def matrix_func(t_X, t_Z):
        xx = tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        dist_xz = xx + tf.transpose(zz) - 2.0 * tf.matmul(t_X, t_Z, transpose_b=True)
        return t_alpha * tf.exp(- 2.0 * t_gamma * (tf.sin(t_period * tf.sqrt(dist_xz + 1.0e-12)) ** 2))

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Periodic',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_matern32_kernel(t_alpha, t_gamma):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        sq_dist_xz = xx + tf.transpose(zz) - t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        sqrt_3 = tf.constant(np.sqrt(3.0), dtype=dtype)
        sqrt_3_dist_xz = sqrt_3 * tf.sqrt(sq_dist_xz + 1.0e-12)
        return t_alpha * (1.0 + sqrt_3_dist_xz) * tf.exp(- sqrt_3_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Matern 3/2',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_matern12_kernel(t_alpha, t_gamma):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        sq_dist_xz = xx + tf.transpose(zz) - t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        dist_xz = tf.sqrt(sq_dist_xz + 1.0e-12)
        return t_alpha * tf.exp(- dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Matern 1/2',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)

def create_linear_kernel(t_alpha):
    def matrix_func(t_X, t_Z):
        return t_alpha * tf.matmul(t_X, t_Z, transpose_b=True)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Linear',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)

def create_gp(t_X, t_Y, t_beta, kernel,
              t_hyper_prior=tf.constant(0.0, dtype=dtype),
              t_extra_prior=None,
              data_noise=None,
              prediction_noise=False,
              predict_exp=False):

    t_N = tf.shape(t_Y)[0]
    t_D = tf.shape(t_Y)[1]
    t_Q = tf.shape(t_X)[1]

    assert (issubclass(type(kernel), Kernel))

    jitter = 1.0e-8

    if data_noise is None:
        t_K_xx = kernel.covar_matrix(t_X, t_X) + ((1.0 / t_beta) + jitter) * tf.eye(t_N, dtype=dtype)
    else:
        # t_K_xx = kernel_function(t_X, t_X) + ((1.0 / t_beta) + jitter) * tf.diag(data_noise)
        t_K_xx = kernel.covar_matrix(t_X, t_X) + 1.0 * tf.diag(data_noise) + jitter * tf.eye(t_N, dtype=dtype)

    t_L_xx = tf.cholesky(t_K_xx, name='L_xx')

    t_log_det = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(t_L_xx)), name='log_det_K_xx')

    t_data_fit = 0.5 * tf.reduce_sum(tf.square(
        tf.matrix_triangular_solve(t_L_xx, t_Y, lower=True)))

    t_noise_prior = 0.5 * tf.square(tf.log(t_beta), name='noise_prior')

    half_log_two_pi = tf.constant(0.5 * np.log(2.0 * np.pi), name='half_log_two_pi', dtype=dtype)

    t_neg_log_likelihood = half_log_two_pi * dtype_convert(t_D) * dtype_convert(t_N) + \
                           0.5 * dtype_convert(t_D) * t_log_det + \
                           t_data_fit + t_hyper_prior + t_noise_prior

    if t_extra_prior != None:
        t_neg_log_likelihood += t_extra_prior

    def create_prediction(tf_input, tf_noise=None, predict_exp=False):
        t_Kinv_Y = tf.matrix_triangular_solve(t_L_xx,
                                              tf.matrix_triangular_solve(t_L_xx, t_Y, lower=True),
                                              lower=True, adjoint=True)
        t_K_x_X = kernel.covar_matrix(tf_input, t_X)
        t_y_mean = tf.matmul(t_K_x_X, t_Kinv_Y)

        if tf_noise is None:
            t_K_x_x_diag = tf.diag_part(kernel.covar_matrix(tf_input, tf_input)) + \
                           (1.0 / t_beta) * tf.ones([tf.shape(tf_input)[0]], dtype=dtype)
        else:
            t_K_x_x_diag = tf.diag_part(kernel.covar_matrix(tf_input, tf_input)) + \
                           tf_noise

        t_y_var = t_K_x_x_diag - tf.reduce_sum(tf.square(
            tf.matrix_triangular_solve(t_L_xx, tf.transpose(t_K_x_X), lower=True)),
            axis=0)
        t_y_var = t_y_var[:, tf.newaxis]

        if predict_exp:
            t_y_mean = tf.exp(t_y_mean)

        # Compute full posterior covariance matrix
        # t_K_X_X = kernel.covar_matrix(t_X, t_X)
        # t_K_X_X_inv = tf.matrix_inverse(t_K_X_X + (1.0 / t_beta) * tf.eye(tf.shape(t_K_X_X)[0], dtype=dtype))
        # t_pred_covar = t_K_x_x + (1.0 / t_beta) * tf.eye(tf.shape(tf_input)[0], dtype=dtype) - \
        #                tf.matmul(tf.matmul(t_K_x_X, t_K_X_X_inv, transpose_a=False), t_K_x_X, transpose_b=True)

        t_K_x_x = kernel.covar_matrix(tf_input, tf_input)
        t_K_XX_inv_K_Xx = tf.cholesky_solve(t_L_xx, tf.transpose(t_K_x_X))
        t_pred_covar = t_K_x_x - tf.matmul(t_K_x_X, t_K_XX_inv_K_Xx) + jitter * tf.eye(tf.shape(tf_input)[0], dtype=dtype)

        return t_y_mean, t_y_var, t_pred_covar

    t_prediction_placeholder = tf.placeholder(dtype=dtype)

    if prediction_noise:
        t_prediction_noise_placeholder = tf.placeholder(dtype=dtype)
    else:
        t_prediction_noise_placeholder = None

    t_prediction_mean, t_prediction_var, t_pred_covar = create_prediction(t_prediction_placeholder, t_prediction_noise_placeholder, predict_exp=predict_exp)

    class GP(object):
        def __init__(self, t_neg_log_likelihood):
            self.name = 'GP'
            self.t_neg_log_likelihood = t_neg_log_likelihood

        def create_prediction(self, tf_input):
            return create_prediction(tf_input)

        @property
        def t_prediction_placeholder(self): return t_prediction_placeholder

        @property
        def t_prediction_noise_placeholder(self): return t_prediction_noise_placeholder

        @property
        def t_prediction_mean(self): return t_prediction_mean

        @property
        def t_prediction_var(self): return t_prediction_var

        @property
        def t_input(self): return t_X

        @property
        def t_output(self): return t_Y

        @property
        def kernel(self): return kernel

        @property
        def kernel_chol(self): return t_L_xx

        @property
        def kernel_matrix(self): return t_K_xx

    gp = GP(t_neg_log_likelihood=t_neg_log_likelihood)

    gp.t_extra_prior = t_extra_prior

    return gp

def create_sparse_gp_fitc(t_X, t_Y, t_beta,
                          kernel,
                          t_inducing_points,
                          t_hyper_prior=tf.constant(0.0, dtype=dtype),
                          t_extra_prior=None, data_noise=None):
    t_N = tf.shape(t_Y)[0]
    t_D = tf.shape(t_Y)[1]
    t_Q = tf.shape(t_X)[1]

    assert (issubclass(type(kernel), Kernel))

    t_M = tf.shape(t_inducing_points)[0]
    t_Z = t_inducing_points

    jitter = 1.0e-8

    t_K_xx_diag = kernel.covar_diag(t_X)
    t_K_ux = kernel.covar_matrix(t_Z, t_X)
    t_K_uu = kernel.covar_matrix(t_Z, t_Z) + jitter * tf.eye(t_M, dtype=dtype)

    t_L_uu = tf.cholesky(t_K_uu, name='L_uu')
    t_V = tf.matrix_triangular_solve(t_L_uu, t_K_ux)

    t_Q_xx_diag = tf.reduce_sum(tf.square(t_V), 0)
    t_Lambda = t_K_xx_diag - t_Q_xx_diag + (1 / t_beta)

    t_B = tf.eye(t_M, dtype=dtype) + tf.matmul(t_V / t_Lambda, t_V, transpose_b=True)
    t_L_B = tf.cholesky(t_B)

    t_beta = t_Y / tf.expand_dims(t_Lambda, 1)
    t_alpha = tf.matmul(t_V, t_beta)
    t_gamma = tf.matrix_triangular_solve(t_L_B, t_alpha, lower=True)

    t_log_det = tf.reduce_sum(tf.log(t_Lambda)) - 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(t_L_B)))
    half_log_two_pi = tf.constant(0.5 * np.log(2.0 * np.pi), name='half_log_two_pi', dtype=dtype)
    
    t_neg_log_likelihood = half_log_two_pi * dtype_convert(t_D) * dtype_convert(t_N) \
                           + 0.5 * dtype_convert(t_D) * t_log_det \
                           + 0.5 * tf.reduce_sum(tf.square(t_Y) / tf.expand_dims(t_Lambda, 1)) \
                           - 0.5 * tf.reduce_sum(tf.square(t_gamma))

    if t_extra_prior != None:
        t_neg_lower_bound += t_extra_prior

    def create_prediction(tf_input):

        t_K_Z_x = kernel.covar_matrix(t_Z, tf_input)
        t_K_x_x_diag = kernel.covar_diag(tf_input)

        t_w = tf.matrix_triangular_solve(t_L_uu, t_K_Z_x, lower=True) 
        t_tmp = tf.matrix_triangular_solve(tf.transpose(t_L_B), t_gamma, lower=False)
        t_y_mean = tf.matmul(t_w, t_tmp, transpose_a=True)

        t_intermediateA = tf.matrix_triangular_solve(t_L_B, t_w, lower=True)
        t_y_var = t_K_x_x_diag - tf.reduce_sum(tf.square(t_w), 0) + tf.reduce_sum(tf.square(t_intermediateA), 0)
        t_y_var = tf.tile(t_y_var[:, None], [1, t_D])

        return t_y_mean, t_y_var

    t_prediction_placeholder = tf.placeholder(dtype=dtype)
    t_prediction_mean, t_prediction_var = create_prediction(t_prediction_placeholder)

    class GP(object):
        def __init__(self, t_neg_log_likelihood):
            self.name = 'SparseGP_FITC'
            self.t_neg_log_likelihood = t_neg_log_likelihood

        def create_prediction(self, tf_input):
            return create_prediction(tf_input)

        @property
        def t_prediction_placeholder(self): return t_prediction_placeholder

        @property
        def t_prediction_mean(self): return t_prediction_mean

        @property
        def t_prediction_var(self): return t_prediction_var

        @property
        def t_input(self): return t_X

        @property
        def t_output(self): return t_Y

    gp = GP(t_neg_log_likelihood=t_neg_log_likelihood)

    gp.t_extra_prior = t_extra_prior

    gp.t_Z = t_Z

    return gp

def create_sparse_gp(t_X, t_Y, t_beta,
                     kernel,
                     t_inducing_points,
                     t_hyper_prior=tf.constant(0.0, dtype=dtype),
                     t_extra_prior=None, data_noise=None):
    t_N = tf.shape(t_Y)[0]
    t_D = tf.shape(t_Y)[1]
    t_Q = tf.shape(t_X)[1]

    assert (issubclass(type(kernel), Kernel))

    t_M = tf.shape(t_inducing_points)[0]
    t_Z = t_inducing_points

    jitter = 1.0e-8

    t_K_uu = kernel.covar_matrix(t_Z, t_Z) + (jitter) * tf.eye(t_M, dtype=dtype)
    
    
    # t_K_uu = kernel.covar_matrix(t_Z, t_Z) + (jitter) * tf.eye(t_M, dtype=dtype)
    t_L_uu = tf.cholesky(t_K_uu, name='L_uu')

    t_K_ux = kernel.covar_matrix(t_Z, t_X) #+ (jitter) * tf.eye(t_N, dtype=dtype)

    t_K_xx_diag = kernel.covar_diag(t_X)

    # Omega = K_xu L_uu^-T [M x N]
    t_OmegaT = tf.matrix_triangular_solve(t_L_uu, t_K_ux, lower=True)

    # C = Omega^T Omega [M x M]
    t_C = tf.matmul(t_OmegaT, t_OmegaT, transpose_b=True)

    t_A = t_C + ((1.0 / t_beta) + jitter) * tf.eye(t_M, dtype=dtype)
    t_L_aa = tf.cholesky(t_A, name='L_aa')

    # Gamma = Y^T Omega
    t_GammaT = tf.matmul(t_OmegaT, t_Y)

    # Psi = Gamma L_aa^-1
    t_PsiT = tf.matrix_triangular_solve(t_L_aa, t_GammaT, lower=True)

    t_yTy = tf.reduce_sum(tf.square(t_Y))

    half_log_two_pi = tf.constant(0.5 * np.log(2.0 * np.pi), name='half_log_two_pi', dtype=dtype)

    log_det_A = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(t_L_aa)))

    t_noise_prior = 0.5 * tf.square(tf.log(t_beta), name='noise_prior')

    t_neg_lower_bound = half_log_two_pi * dtype_convert(t_D) * dtype_convert(t_N) \
                        - 0.5 * dtype_convert(t_D) * dtype_convert(t_N - t_M) * tf.log(t_beta) \
                        + 0.5 * dtype_convert(t_D) * log_det_A \
                        + 0.5 * t_beta * t_yTy \
                        - 0.5 * t_beta * tf.reduce_sum(tf.square(t_PsiT)) \
                        + 0.5 * t_beta * (tf.reduce_sum(t_K_xx_diag) - tf.trace(t_C)) \
                        + t_hyper_prior \
                        + t_noise_prior

    if t_extra_prior != None:
        t_neg_lower_bound += t_extra_prior

    def create_prediction(tf_input):

        t_AInv = tf.cholesky_solve(t_L_aa, tf.eye(t_M, dtype=dtype))

        t_PredictAlpha = (1.0/t_beta) * tf.matrix_triangular_solve(t_L_uu,
                                                            tf.matmul(t_AInv, t_GammaT),
                                                            lower=True, adjoint=True)

        t_K_x_Z = kernel.covar_matrix(tf_input, t_Z)

        t_y_mean = t_beta * tf.matmul(t_K_x_Z, t_PredictAlpha)

        t_K_x_x_diag = kernel.covar_diag(tf_input)

        t_L_uuInv_K_Z_x = tf.matrix_triangular_solve(t_L_uu, tf.transpose(t_K_x_Z), lower=True)

        t_G = (1.0/t_beta) * t_AInv - tf.eye(t_M, dtype=dtype)

        t_y_var = t_K_x_x_diag \
                  + tf.reduce_sum(t_L_uuInv_K_Z_x * tf.matmul(t_G, t_L_uuInv_K_Z_x), axis=0) \
                  + (1.0 / t_beta) * tf.ones([tf.shape(tf_input)[0]], dtype=dtype)

        t_y_var = t_y_var[:, tf.newaxis]

        return t_y_mean, t_y_var

    t_prediction_placeholder = tf.placeholder(dtype=dtype)
    t_prediction_mean, t_prediction_var = create_prediction(t_prediction_placeholder)

    class GP(object):
        def __init__(self, t_neg_log_likelihood):
            self.name = 'SparseGP'
            self.t_neg_log_likelihood = t_neg_log_likelihood

        def create_prediction(self, tf_input):
            return create_prediction(tf_input)

        @property
        def t_prediction_placeholder(self): return t_prediction_placeholder

        @property
        def t_prediction_mean(self): return t_prediction_mean

        @property
        def t_prediction_var(self): return t_prediction_var

        @property
        def t_input(self): return t_X

        @property
        def t_output(self): return t_Y

    gp = GP(t_neg_log_likelihood=t_neg_lower_bound)

    gp.t_extra_prior = t_extra_prior

    gp.t_Z = t_Z

    return gp


def get_scaled_limits(X, scale_factor=1.1, axis=0):
    Xmin = np.min(X, axis=axis)
    Xmax = np.max(X, axis=axis)

    Q = len(Xmin)

    Xdim = Xmax - Xmin

    nXdim = np.max(Xdim) * np.ones([Q])

    Xmid = 0.5 * (Xmin + Xmax)

    Xmin = Xmid - scale_factor * 0.5 * nXdim
    Xmax = Xmid + scale_factor * 0.5 * nXdim

    return Xmin, Xmax


# Helper function for GP creation..
def create_standard_gp(t_X, t_Y, t_extra_prior=None, data_noise=None, predict_exp=False):
    t_alpha = positive_variable(1.0)
    t_gamma = positive_variable(1.0)
    t_beta = positive_variable(1.0)
    kernel = create_squared_exp_kernel(t_alpha=t_alpha, t_gamma=t_gamma)
    t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha)) + tf.square(tf.log(t_gamma)))

    gp = create_gp(t_X, t_Y, t_beta, kernel=kernel, t_hyper_prior=t_hyper_prior,
                   t_extra_prior=t_extra_prior, data_noise=data_noise, predict_exp=predict_exp)

    gp.hyperparams = (t_alpha, t_gamma, t_beta)

    return gp


# Helper function for SparseGP creation..
def create_standard_sparse_gp(t_X, t_Y, num_inducing_points, num_input_dims, t_extra_prior=None):
    t_alpha = positive_variable(1.0)
    t_gamma = positive_variable(1.0)
    t_beta = positive_variable(1.0)
    kernel = create_squared_exp_kernel(t_alpha=t_alpha, t_gamma=t_gamma)
    t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha)) + tf.square(tf.log(t_gamma)))

    t_inducing_points = real_variable(init_linspace_inducing_points(num_inducing_points, num_input_dims))

    gp = create_sparse_gp(t_X, t_Y, t_beta,
                          kernel=kernel,
                          t_inducing_points=t_inducing_points,
                          t_hyper_prior=t_hyper_prior,
                          t_extra_prior=t_extra_prior)

    gp.hyperparams = (t_alpha, t_gamma, t_beta)

    return gp


def init_linspace_inducing_points(num_inducing_points, num_input_dims, low=-1.0, high=1.0):
    return np.repeat(np.linspace(low, high, num_inducing_points)[:, np.newaxis],
                                 num_input_dims, axis=1)
   
def tf_pdist2(X, Y):
    xx = tf.reduce_sum(X * X, axis=1, keepdims=True)
    yy = tf.reduce_sum(Y * Y, axis=1, keepdims=True)
    return xx + tf.transpose(yy) - 2.0 * tf.matmul(X, Y, transpose_b=True)

def create_DP_clustering(t_X, N, Q, T=20,
                        init='random',
                        init_data=None,
                        mixture_sigma=None,
                        lambda_sigma=None,
                        alpha=None,
                        sigma_s2=0.1,
                        lambda_s2=0.1,
                        alpha_s1=1.0,
                        alpha_s2=1.0,
                        alpha_in_bound=False,
                        lambda_in_bound=False,
                        alpha_a=2.0,
                        alpha_b=1.0,
                        lambda_a=2.0,
                        lambda_b=1.0):
    tf_pi = tf.constant(np.pi, dtype=dtype)
    tf_Q = tf.constant(Q, dtype=dtype)

    alpha_a = tf.constant(alpha_a, dtype=dtype)
    alpha_b = tf.constant(alpha_b, dtype=dtype)

    lambda_a = tf.constant(lambda_a, dtype=dtype)
    lambda_b = tf.constant(lambda_b, dtype=dtype)

    def create_variational_parameters():
        # parameters of beta-distributed q(V) (gamma)
        # gamma has shape (T-1, 2): two params for T-1 beta distributions
        # log_gamma_init_1 = np.zeros(T-1)
        # log_gamma_init_2 = np.linspace(-1, 1, T-1) # np.random.randn(T-1) * 0
        # t_gamma_1 = positive_variable(np.exp(log_gamma_init_1))
        # t_gamma_2 = positive_variable(np.exp(log_gamma_init_2))

        t_gamma_1 = positive_variable(np.ones(T - 1) * alpha_a / alpha_b)
        t_gamma_2 = positive_variable(np.ones(T - 1) * (T - 2) * alpha_a / alpha_b)

        if init == 'random':
            # parameters for mixture components q(eta) (gaussian means and the same variance for all mixtures)
            t_tau = real_variable(np.random.randn(T, Q))

            # parameters cluster assignments q(z) (multinomial)
            t_phi_unnormalised = real_variable(np.random.randn(N, T))

            # phi_init = np.zeros((N, T))
            # phi_init[:,0] = 1
            # t_phi_unnormalised = real_variable(phi_init)
            t_phi = tf.nn.softmax(t_phi_unnormalised, axis=-1)

        elif init == 'kmeans':
            assert init_data is not None
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=T)
            init_clusters = kmeans.fit_predict(init_data)

            # parameters for mixture components q(eta) (gaussian means and the same variance for all mixtures)
            t_tau = real_variable(kmeans.cluster_centers_)

            # parameters cluster assignments q(z) (multinomial)
            phi_unnormalised_init = np.zeros((N, T))
            phi_unnormalised_init[(range(N), init_clusters)] = 1
            t_phi_unnormalised = real_variable(phi_unnormalised_init)
            t_phi = tf.nn.softmax(t_phi_unnormalised, axis=-1)
        else:
            raise Exception('init must be either "random" or "kmeans"')

        alpha_w1 = positive_variable(alpha_a)
        alpha_w2 = positive_variable(alpha_b)

        lambda_w1 = positive_variable(lambda_a)
        lambda_w2 = positive_variable(lambda_b)

        return t_gamma_1, t_gamma_2, t_tau, t_phi, alpha_w1, alpha_w2, lambda_w1, lambda_w2

    ## hyperparameters

    ## t_alpha is a scaling parameter of the DP, allowing to adjust the number of clusters (the smaller it is the less clusters are found).

    ## t_mixture_sigma is the variance of each mixture component, also implicitely controlling the number of clusters, as the small value
    ## of this parameter means many mixture components (e.g. Gaussians), each explaning a few data points, while the large value might
    ## allow to explain all the data with a single wide mixture component 

    ## So t_alpha and t_mixture_sigma seem to be correlated in their effect on the final solution, so we fix t_alpha and optimise only
    ## t_mixture_sigma.

    ## t_lambda_mean and t_lambda_sigma are the parameters of the prior distribution for the locations of the mixture components and they
    ## don't seem to influence the solution much, therefore we set them to fixed values.

    if alpha is None:
        t_alpha = positive_variable(alpha_s1 / alpha_s2) 
    else:
        if type(alpha) == tf.Tensor:
            t_alpha = alpha
        else:
            t_alpha = tf.constant(alpha, dtype=dtype)

    t_lambda_mean = tf.constant([[0.0] * Q], dtype=dtype)

    if lambda_sigma is None:
        t_lambda_sigma = positive_variable(2.0)
    else:
        if type(lambda_sigma) == tf.Tensor:
            t_lambda_sigma = lambda_sigma
        else:
            t_lambda_sigma = tf.constant(lambda_sigma, dtype=dtype)

    if mixture_sigma is None:
        t_mixture_sigma = positive_variable(.1)
    else:
        if type(mixture_sigma) == tf.Tensor:
            t_mixture_sigma = mixture_sigma
        else:
            t_mixture_sigma = tf.constant(mixture_sigma, dtype=dtype)

    t_gamma_1, t_gamma_2, t_tau, t_phi, alpha_w1, alpha_w2, lambda_w1, lambda_w2 = create_variational_parameters()

    t_E_logVt = tf.digamma(t_gamma_1) - tf.digamma(t_gamma_1 + t_gamma_2)
    t_E_log1mVt = tf.digamma(t_gamma_2) - tf.digamma(t_gamma_1 + t_gamma_2)

    if not alpha_in_bound:
        t_E_pV = (t_alpha - 1) * tf.reduce_sum(t_E_log1mVt) - (T - 1) * (tf.lgamma(t_alpha) - tf.lgamma(1 + t_alpha))
    else:
        # t_E_pV = (alpha_w1 / alpha_w2 - 1) * tf.reduce_sum(t_E_log1mVt) + (T + alpha_a - 2) * (tf.digamma(alpha_w1) - tf.log(alpha_w2)) - \
        #          alpha_b * alpha_w1 / alpha_w2 - tf.lgamma(alpha_a) + alpha_a * tf.log(alpha_b)
        t_E_pV = (alpha_w1 / alpha_w2 - 1) * tf.reduce_sum(t_E_log1mVt) + \
                 (T - 1 + alpha_a) * (tf.digamma(alpha_w1) - tf.log(alpha_w2)) - \
                 alpha_w1 * tf.digamma(alpha_w1) + tf.lgamma(alpha_w1) - tf.lgamma(alpha_a) + \
                 alpha_a * tf.log(alpha_b) - (alpha_w1 / alpha_w2) * (alpha_b - alpha_w2)

    if not lambda_in_bound:
        t_E_pEta = tf.reduce_sum(-0.5 * (t_tau - t_lambda_mean)**2 / t_lambda_sigma, axis=1) - 0.5 * tf_Q * tf.log(2 * tf_pi * t_lambda_sigma)
        t_E_pEta = tf.reduce_sum(t_E_pEta)
    else:
        t_E_pEta = -(lambda_w1 / lambda_w2) * (0.5 * tf.reduce_sum((t_tau - t_lambda_mean)**2) + lambda_b - lambda_w2) + \
                    (0.5 * tf_Q * T + lambda_a) * (tf.digamma(lambda_w1) - tf.log(lambda_w2)) - \
                    0.5 * tf_Q * T * tf.log(2 * tf_pi) - lambda_w1 * tf.digamma(lambda_w1) + \
                    tf.lgamma(lambda_w1) - tf.lgamma(lambda_a) + lambda_a * tf.log(lambda_b)

    t_E_pZV = tf.reduce_sum(t_E_logVt * t_phi[:,:-1] + t_E_log1mVt * tf.cumsum(t_phi, exclusive=True, reverse=True , axis=1)[:,:-1], axis=1)
    t_E_pZV = tf.reduce_sum(t_E_pZV)
    
    t_E_pxZ = tf.reduce_sum(t_phi * (-0.5 * tf_pdist2(t_X, t_tau) / t_mixture_sigma - 0.5 * tf_Q * tf.log(2 * tf_pi * t_mixture_sigma)), axis=1)
    t_E_pxZ = tf.reduce_sum(t_E_pxZ)

    t_E_qV = (t_gamma_1 - 1) * t_E_logVt + (t_gamma_2 - 1) * t_E_log1mVt - \
             (tf.lgamma(t_gamma_1) + tf.lgamma(t_gamma_2) - tf.lgamma(t_gamma_1 + t_gamma_2))
    t_E_qV = tf.reduce_sum(t_E_qV)

    t_E_qZ = tf.reduce_sum(t_phi * tf.log(t_phi))
    
    t_lower_bound = t_E_pV + t_E_pEta + t_E_pZV + t_E_pxZ - t_E_qV - t_E_qZ

    # add priors on hyperparameters

    # parameters of gamma priors
    alpha_s1 = tf.constant(alpha_s1, dtype=dtype)
    alpha_s2 = tf.constant(alpha_s2, dtype=dtype)
    
    lambda_s1 = tf.constant(1.0, dtype=dtype)
    lambda_s2 = tf.constant(lambda_s2, dtype=dtype)

    sigma_s1 = tf.constant(1.0, dtype=dtype)
    sigma_s2 = tf.constant(sigma_s2, dtype=dtype)
    
    prior_alpha = alpha_s1 * tf.log(alpha_s2) - tf.lgamma(alpha_s1) + (alpha_s1 - 1) * tf.log(t_alpha) - alpha_s2 * t_alpha
    prior_lambda_mean = tf.reduce_sum(-0.5 * t_lambda_mean**2)
    prior_lambda_sigma = lambda_s1 * tf.log(lambda_s2) - tf.lgamma(lambda_s1) + tf.reduce_sum((lambda_s1 - 1) * tf.log(t_lambda_sigma) - lambda_s2 * t_lambda_sigma)
    prior_mixture_sigma = sigma_s1 * tf.log(sigma_s2) - tf.lgamma(sigma_s1) + (sigma_s1 - 1) * tf.log(t_mixture_sigma) - sigma_s2 * t_mixture_sigma

    if not alpha_in_bound and not lambda_in_bound:
        t_lower_bound = t_lower_bound + prior_alpha + prior_lambda_sigma
    elif alpha_in_bound and not lambda_in_bound:
        t_lower_bound = t_lower_bound + prior_lambda_sigma
    elif not alpha_in_bound and lambda_in_bound:
        t_lower_bound = t_lower_bound + prior_alpha

    t_neg_lower_bound = -t_lower_bound

    return t_neg_lower_bound, t_tau, t_phi, t_mixture_sigma, t_alpha, t_gamma_1, t_gamma_2, t_lambda_sigma, alpha_w1, alpha_w2, lambda_w1, lambda_w2

def create_BMM_clustering(t_X, N, Q, T=20,
                          alpha=None,
                          mixture_sigma=None,
                          lambda_sigma=None,
                          lambda_s1=1.0,
                          lambda_s2=0.1,
                          alpha_s1=1.0,
                          alpha_s2=1.0):
    tf_pi = tf.constant(np.pi, dtype=dtype)
    tf_Q = tf.constant(Q, dtype=dtype)

    t_lambda_mean = tf.constant([[0.0] * Q], dtype=dtype)

    if alpha is None:
        t_alpha = positive_variable(alpha_s1 / alpha_s2) 
    else:
        if type(alpha) == tf.Tensor:
            t_alpha = alpha
        else:
            t_alpha = tf.constant(alpha, dtype=dtype)

    if lambda_sigma is None:
        t_lambda_sigma = positive_variable(2.0)
    else:
        if type(lambda_sigma) == tf.Tensor:
            t_lambda_sigma = lambda_sigma
        else:
            t_lambda_sigma = tf.constant(lambda_sigma, dtype=dtype)

    if mixture_sigma is None:
        t_mixture_sigma = positive_variable(.1)
    else:
        if type(mixture_sigma) == tf.Tensor:
            t_mixture_sigma = mixture_sigma
        else:
            t_mixture_sigma = tf.constant(mixture_sigma, dtype=dtype)

    # t_tau = real_variable(np.random.randn(T, Q))

    # t_tau_coeffs = tf.nn.softmax(real_variable(np.random.randn(T, N)), axis=1)
    t_tau_coeffs = tf.nn.sigmoid(real_variable(np.random.randn(T, N)))
    t_tau = tf.matmul(t_tau_coeffs, t_X)

    t_gamma = positive_variable((alpha_s1 / alpha_s2) * np.ones(T))
    t_gamma_sum = tf.reduce_sum(t_gamma)

    t_phi_unnormalised = real_variable(np.random.randn(N, T))
    t_phi = tf.nn.softmax(t_phi_unnormalised, axis=-1)

    t_E_pxZ = tf.reduce_sum(t_phi * (-0.5 * tf_pdist2(t_X, t_tau) / t_mixture_sigma - 0.5 * tf_Q * tf.log(2 * tf_pi * t_mixture_sigma)), axis=1)
    t_E_pxZ = tf.reduce_sum(t_E_pxZ)

    t_E_pZpi = tf.reduce_sum(t_phi * (tf.digamma(t_gamma) - tf.digamma(t_gamma_sum)), axis=1)
    t_E_pZpi = tf.reduce_sum(t_E_pZpi)

    t_E_pEmu = tf.reduce_sum(-0.5 * (t_tau - t_lambda_mean)**2 / t_lambda_sigma, axis=1) - 0.5 * tf_Q * tf.log(2 * tf_pi * t_lambda_sigma)
    t_E_pEmu = tf.reduce_sum(t_E_pEmu)

    t_E_ppi = (t_alpha - 1) * (tf.reduce_sum(tf.digamma(t_gamma)) - T * tf.digamma(t_gamma_sum)) - \
              T * tf.lgamma(t_alpha) + tf.lgamma(T * t_alpha)

    t_E_qZ = tf.reduce_sum(t_phi * tf.log(t_phi))

    t_E_qpi = tf.lgamma(t_gamma_sum) - tf.reduce_sum(tf.lgamma(t_gamma)) - (t_gamma_sum - T) * tf.digamma(t_gamma_sum) + \
              tf.reduce_sum((t_gamma - 1) * tf.digamma(t_gamma))

    t_lower_bound = t_E_pxZ + t_E_pZpi + t_E_pEmu + t_E_ppi - t_E_qZ - t_E_qpi

    t_alpha_s1 = tf.constant(alpha_s1, dtype=dtype)
    t_alpha_s2 = tf.constant(alpha_s2, dtype=dtype)
    t_lambda_s1 = tf.constant(lambda_s1, dtype=dtype)
    t_lambda_s2 = tf.constant(lambda_s2, dtype=dtype)

    t_alpha_prior = t_alpha_s1 * tf.log(t_alpha_s2) + (t_alpha_s1 - 1) * tf.log(t_alpha) - \
                    t_alpha_s2 * t_alpha - tf.lgamma(t_alpha_s1)
        
    t_lambda_prior = t_lambda_s1 * tf.log(t_lambda_s2) + (t_lambda_s1 - 1) * tf.log(t_lambda_sigma) - \
                     t_lambda_s2 * t_lambda_sigma - tf.lgamma(t_lambda_s1)

    # t_tau_coeffs_prior = tf.reduce_sum(t_tau_coeffs * tf.log(t_tau_coeffs))
    t_tau_coeffs_prior = -tf.reduce_sum(t_tau_coeffs)

    t_lower_bound = t_lower_bound + t_alpha_prior + t_lambda_prior + t_tau_coeffs_prior

    t_neg_lower_bound = -t_lower_bound

    return t_neg_lower_bound, t_tau, t_phi, t_gamma, t_mixture_sigma, t_alpha, t_lambda_sigma
    
def create_GPLVM_alignment(t_Ys_aligned,
                           y_ref,
                           N_seq,
                           N_samples,
                           num_output_dims,
                           Q=2):
    y_ref_multidim = y_ref.reshape([N_seq, N_samples * num_output_dims])

    X_aligned_init = standard_pca_initialisation(y_ref_multidim, Q)
    t_Xs_aligned = real_variable(X_aligned_init)

    # GPLVM std manifold prior..
    t_gplvm_prior = 0.01*tf.reduce_sum(tf.square(t_Xs_aligned))
    
    # The alignment GP that clusters the aligned estimates from the other GPs..
    alignment_gp = create_standard_gp(t_Xs_aligned, t_Ys_aligned,
                                      t_extra_prior=t_gplvm_prior)

    return t_Xs_aligned, alignment_gp

def mgp_S(s, l, tf_pi):
    return tf.sqrt(2 * tf_pi * l**2) * tf.exp(-2 * tf_pi**2 * l**2 * s**2)

def mgp_log_S(s, l, tf_pi):
    return 0.5 * tf.log(2 * tf_pi * l**2) - 2 * tf_pi**2 * l**2 * s**2

def mgp_generate_sqrt_lambdas(J, L):
    return np.arange(1, J + 1) * np.pi / (2 * L)

def mgp_generate_psi(x, J, L):
    sqrt_lambdas = mgp_generate_sqrt_lambdas(J, L)
    gamma_plus = 2 * sqrt_lambdas
    psi_diag = (x + L) / (2 * L) - np.sin(gamma_plus * (x + L)) / (2 * L * gamma_plus)
    psi = np.diag(psi_diag)
    
    for i in range(J):
        for j in range(i + 1, J):
            gamma_minus_ij = sqrt_lambdas[i] - sqrt_lambdas[j]
            gamma_plus_ij  = sqrt_lambdas[i] + sqrt_lambdas[j]
            psi[i,j] = np.sin(gamma_minus_ij * (x + L)) / (2 * L * gamma_minus_ij) - \
                       np.sin(gamma_plus_ij * (x + L)) / (2 * L * gamma_plus_ij)
            psi[j,i] = psi[i,j]
    return psi

def create_monotonic_gp_variational(t_X, t_Y, N, J, L, t_beta, t_l):
    tf_pi = tf.constant(np.pi, dtype=dtype)
    psis = np.stack([mgp_generate_psi(x, J, L) for x in t_X]) # replace that with a tensorflow function for psi
    t_psi = tf.constant(psis, dtype=dtype)

    # Create variational variables

    t_theta_0 = tf.constant(-L, dtype=dtype)
    t_eta_0 = tf.constant(1.0, dtype=dtype)

    t_mu_0 = real_variable(0.0)
    t_sigma_0 = positive_variable(1.0)

    t_mus = real_variable(np.random.randn(J))
    t_sigmas = positive_variable(np.ones(J))

    t_Sigma = tf.tile(tf.expand_dims(tf.diag(t_sigmas), 0), [N, 1, 1])

    tf_sqrt_lambdas = tf.constant(mgp_generate_sqrt_lambdas(J, L), dtype=dtype)
    tf_S_lambdas = mgp_S(tf_sqrt_lambdas, t_l, tf_pi)
    tf_log_S_lambdas = mgp_log_S(tf_sqrt_lambdas, t_l, tf_pi)

    # Compute the likelihood bound

    t_Tr = tf.trace(tf.matmul(t_psi, t_Sigma))

    t_Nmus = tf.tile(tf.reshape(t_mus, [1, J, 1]), [N, 1, 1])
    t_apa = tf.squeeze(tf.matmul(tf.transpose(tf.matmul(t_psi, t_Nmus), [0, 2, 1]), t_Nmus))

    t_psp = tf.matmul(tf.matmul(t_psi, t_Sigma), t_psi)
    t_psps = tf.trace(tf.matmul(t_psp, t_Sigma))
    t_psp_mu = tf.squeeze(tf.matmul(tf.transpose(tf.matmul(t_psp, t_Nmus), [0, 2, 1]), t_Nmus))

    t_E_sq_diff = t_Y**2 - 2 * t_Y * (t_mu_0 + t_Tr + t_apa) + \
                  t_sigma_0 + t_mu_0**2 + 2 * t_mu_0 * (t_Tr + t_apa) + \
                  2 * t_psps + 4 * t_psp_mu + (t_Tr + t_apa) ** 2

    t_Eqp = -(N / 2) * (tf.log(2 * tf_pi) - tf.log(t_beta)) - \
        (t_beta / 2) * tf.reduce_sum(t_E_sq_diff)

    t_KL_alpha = t_mus**2 / (2 * tf_S_lambdas) + \
                 0.5 * (t_sigmas / tf_S_lambdas - 1 - tf.log(t_sigmas) + tf_log_S_lambdas)
    t_KL_alpha = tf.reduce_sum(t_KL_alpha)

    # t_KL_f0 = t_mu_0**2 / 2 + 0.5 * (t_sigma_0 - 1 - tf.log(t_sigma_0))
    t_KL_f0 = (t_mu_0 - t_theta_0)**2 / (2 * t_eta_0) + \
              0.5 * (t_sigma_0 / t_eta_0 - 1 - tf.log(t_sigma_0) + tf.log(t_eta_0))

    t_lower_bound = t_Eqp - t_KL_alpha - t_KL_f0
    t_neg_lower_bound = -t_lower_bound

    return t_neg_lower_bound, t_mu_0, t_mus, t_psi

def create_monotonic_gp_point_estimate(t_X, N, J, L):
    tf_pi = tf.constant(np.pi, dtype=dtype)
    psis = np.stack([mgp_generate_psi(x, J, L) for x in t_X]) # replace that with a tensorflow function for psi
    t_psi = tf.constant(psis, dtype=dtype)

    # t_alpha = real_variable(np.random.randn(J))
    alpha_init = np.array([ 1.63530603, -0.36321679, -0.30773932, -0.22866863, -0.21394494])
    t_alpha = real_variable(alpha_init)
    # t_alpha = tf.constant(np.array([ 1.63530603, -0.36321679, -0.30773932, -0.22866863, -0.21394494]), dtype=dtype)
    # t_f0 = real_variable(-L)
    f0_init = -1.2436
    t_f0 = real_variable(f0_init)
    # t_f0 = tf.constant(-1.2436, dtype=dtype)

    t_weights_prior = tf.reduce_sum((t_alpha - alpha_init)**2) + (t_f0 - f0_init)**2

    t_Nalphas = tf.tile(tf.reshape(t_alpha, [1, J, 1]), [N, 1, 1])
    t_apa = tf.squeeze(tf.matmul(tf.transpose(tf.matmul(t_psi, t_Nalphas), [0, 2, 1]), t_Nalphas))
    t_output = t_apa + t_f0

    return t_output, t_alpha, t_f0, t_weights_prior
import os
import glob
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture as sm

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

try:
    from .Alignment_Utilities import create_sparse_gp, create_squared_exp_kernel, tf_pdist2
except ImportError:
    from Alignment_Utilities import create_sparse_gp, create_squared_exp_kernel, tf_pdist2

pickle_files = {'sparse_init': '/gpfs01/bethge/home/iustyuzh/cnn-sys-ident/dense_readout_sparse_init.pkl',
                'rand_init': '/gpfs01/bethge/home/iustyuzh/cnn-sys-ident/dense_readout_rand_init.pkl',
                'rand_init_train_core': '/gpfs01/bethge/home/iustyuzh/cnn-sys-ident/dense_readout_rand_init_train_core.pkl',
                'no_init': '/gpfs01/bethge/home/iustyuzh/cnn-sys-ident/dense_readout_no_init.pkl'}

def get_val_test_scores(model_type):
    filename = pickle_files[model_type]
    with open(filename, 'rb') as f:
        losses = pickle.load(f)

    reg_weights = []
    val_loss = []
    test_corr = []
    Ws = []

    for rw, (vl, tc, W) in losses.items():
        reg_weights.append(rw)
        val_loss.append(vl)
        test_corr.append(tc)
        Ws.append(W)

    test_corr = np.array(test_corr)
    if np.ndim(test_corr) == 1:
        test_corr = np.reshape(test_corr, (-1, 1))

    return np.array(reg_weights), \
           np.array(val_loss), \
           np.array(test_corr), \
           np.array(Ws)

def compute_entropy(W):
    W = np.abs(W) / np.sum(np.abs(W), axis=2, keepdims=True)
    entropy = -np.sum(W * np.log(W + 1e-10), axis=2)
    return np.mean(entropy, axis=1)

def compute_l1_over_l2(W):
    ratio = np.linalg.norm(W, axis=2, ord=1) / np.linalg.norm(W, axis=2, ord=2)
    return np.mean(ratio, axis=1)

def fit_mm(W, n_components, mm_type, alpha=None):
    if mm_type == 'em':
        gmm = sm.GaussianMixture(n_components=n_components)
    elif mm_type == 'var':
        assert alpha is not None
        gmm = sm.BayesianGaussianMixture(n_components=n_components,
                                         weight_concentration_prior_type='dirichlet_distribution',
                                         weight_concentration_prior=alpha)
    else:
        raise Exception('mm_type must be in ["em", "var"]')

    gmm.fit(W)
    return gmm

def compute_gmm_scores(gmm, W, mm_type):
    scores = {'params': gmm.get_params(),
              'pred': gmm.predict(W),
              'probs': gmm.predict_proba(W),
              'score': gmm.score(W),
              'score_samples': gmm.score_samples(W),
              'lower_bound': gmm.lower_bound_,
              'weights': gmm.weights_,
              'converged': gmm.converged_,
              'means': gmm.means_}

    if mm_type == 'em':
        scores.update({'aic': gmm.aic(W), 'bic': gmm.bic(W)})

    return scores

def pickle_file(file_name, content):
    with open(file_name, 'wb') as f:
        pickle.dump(content, f)

def marginalise_orientations_energy(W, n_features, normalise=False, normalise_weights=None):
    n_orientations = W.shape[1] // n_features

    if normalise_weights is not None:
        W = W * np.reshape(normalise_weights, (-1, 1))
    elif normalise:
        W = W / np.linalg.norm(W, axis=1, keepdims=True)

    marginalised_W = []
    for f in range(n_features):
        orientations_idx = np.arange(f, W.shape[1], n_features)
        W_f = np.sum(W[:, orientations_idx]**2, axis=1)
        marginalised_W.append(W_f)

    return np.stack(marginalised_W, axis=-1)

def normalise_W(W, normalise=False, normalise_weights=None):
    if normalise_weights is not None:
        W = W * np.reshape(normalise_weights, (-1, 1))
    elif normalise:
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
    return W

def run_gmm(fitted_results,
            save_dir,
            save_file,
            components,
            alphas,
            train_idx,
            test_idx,
            n_best=10,
            orientations_energy=False,
            normalise=False,
            normalise_weights=None):

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, save_file + '.pkl')

    try:
        with open(save_file, 'rb') as f:
            fitted_GMMs = pickle.load(f)
    except:
        fitted_GMMs = {}

    start_time = time.time()
    fitted_results = sorted(list(zip(*fitted_results)), key=lambda e: e[1])[:n_best]
    for i, (rw, val_loss, test_corr, W) in enumerate(fitted_results):
        print('{} / {}: '.format(i + 1, n_best), end=' ')

        W_train = W[train_idx]
        W_test = W[test_idx]

        if orientations_energy:
            if normalise_weights is not None:
                nw_train = normalise_weights[rw][train_idx]
                nw_test = normalise_weights[rw][test_idx]
            else:
                nw_train = None
                nw_test = None
            W_train = marginalise_orientations_energy(W_train, 16, normalise=normalise, normalise_weights=nw_train)
            W_test = marginalise_orientations_energy(W_test, 16, normalise=normalise, normalise_weights=nw_test)
        elif normalise:
            W_train = normalise_W(W_train, normalise=normalise)
            W_test = normalise_W(W_test, normalise=normalise)

        if not rw in fitted_GMMs:
            fitted_GMMs[rw] = {'val_loss': val_loss,
                               'test_corr': test_corr,
                               'GMMs_EM': {},
                               'GMMs_VI': {}}

        for n_components in components:
            print(n_components, end=' ')
            if n_components in fitted_GMMs[rw]['GMMs_EM']:
                continue
            gmm = fit_mm(W_train, n_components, 'em')
            fitted_GMMs[rw]['GMMs_EM'][n_components] = \
                compute_gmm_scores(gmm, W_test, 'em')
            pickle_file(save_file, fitted_GMMs)

        print('')
        print(' ' * 9, end='')

        for n_components in components:
            print(n_components, end=' ')
            if n_components not in fitted_GMMs[rw]['GMMs_VI']:
                fitted_GMMs[rw]['GMMs_VI'][n_components] = {}
            for alpha in alphas:
                if alpha in fitted_GMMs[rw]['GMMs_VI'][n_components]:
                    continue
                gmm = fit_mm(W_train, n_components, 'var', alpha)
                fitted_GMMs[rw]['GMMs_VI'][n_components][alpha] = \
                    compute_gmm_scores(gmm, W_test, 'var') 
                pickle_file(save_file, fitted_GMMs)

        print('')

def run_oriented_gmm(
        fitted_results,
        save_dir,
        save_file,
        components,
        train_idx,
        test_idx,
        n_best=10,
        normalise=False,
        normalise_weights=None):

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, save_file + '.pkl')

    try:
        with open(save_file, 'rb') as f:
            fitted_GMMs = pickle.load(f)
    except:
        fitted_GMMs = {}

    start_time = time.time()
    fitted_results = sorted(list(zip(*fitted_results)), key=lambda e: e[1])[:n_best]
    for i, (rw, val_loss, test_corr, W) in enumerate(fitted_results):
        print('\t\t\t\t*** {} / {}: ***'.format(i + 1, n_best), end='\n\n')

        W_train = W[train_idx]
        W_test = W[test_idx]

        if normalise_weights is not None:
            nw_train = normalise_weights[rw][train_idx]
            nw_test = normalise_weights[rw][test_idx]
        else:
            nw_train = None
            nw_test = None
        W_train = normalise_W(W_train, normalise=normalise, normalise_weights=nw_train)
        W_test = normalise_W(W_test, normalise=normalise, normalise_weights=nw_test)

        if not rw in fitted_GMMs:
            fitted_GMMs[rw] = {'val_loss': val_loss,
                               'test_corr': test_corr,
                               'GMMs_oriented': {}}

        for n_components in components:
            print('\t\t\t*** n_components = {} ***'.format(n_components))
            if n_components in fitted_GMMs[rw]['GMMs_oriented']:
                continue

            tf.reset_default_graph()
            
            oriented_gmm_train = OrientationsGMM(
                log_dir=save_dir + '_checkpoints',
                N=W_train.shape[0],
                K=n_components,
                num_features=16,
                num_orientations=8,
                alpha=100,
                orientations_reg_coeff=1e6,
                covar_init_std=3 * np.max(np.abs(W_train)),
                jitter=0)

            oriented_gmm_train.fit(
                W_train,
                max_iter=100000,
                eval_steps=100,
                init_lr=1e-1)

            train_results = oriented_gmm_train.get_parameters('train')

            print('')

            oriented_gmm_test = OrientationsGMM(
                log_dir=save_dir + '_checkpoints',
                N=W_test.shape[0],
                K=n_components,
                num_features=16,
                num_orientations=8,
                alpha=100,
                orientations_reg_coeff=1e6,
                covar_init_std=3 * np.max(np.abs(W_test)),
                jitter=0,
                means=train_results['train_means'],
                covars=train_results['train_covars'])

            oriented_gmm_test.fit(
                W_test,
                max_iter=100000,
                eval_steps=100,
                init_lr=1e-1)

            test_results = oriented_gmm_test.get_parameters('test')
            test_results.update(train_results)

            fitted_GMMs[rw]['GMMs_oriented'][n_components] = test_results
            pickle_file(save_file, fitted_GMMs)

        print('')

def run_rotations_GPLVM(
        fitted_results,
        save_dir,
        save_file,
        latent_prior_coeffs,
        max_temps,
        num_inducing_points,
        n_best=10,
        normalise=False,
        normalise_weights=None):

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, save_file + '.pkl')

    try:
        with open(save_file, 'rb') as f:
            fitted_GPLVMs = pickle.load(f)
    except:
        fitted_GPLVMs = {}

    fitted_results = sorted(list(zip(*fitted_results)), key=lambda e: e[1])[:n_best]
    for i, (rw, val_loss, test_corr, W) in enumerate(fitted_results):
        print('\t\t\t\t*** {} / {}: ***'.format(i + 1, n_best), end='\n\n')

        if normalise_weights is not None:
            nw = normalise_weights[rw]
        else:
            nw = None
        W = normalise_W(W, normalise=normalise, normalise_weights=nw)

        W = np.transpose(np.reshape(W, (W.shape[0], 8, 16)), (0, 2, 1))

        if not rw in fitted_GPLVMs:
            fitted_GPLVMs[rw] = {'val_loss': val_loss,
                                 'test_corr': test_corr,
                                 'rotated_GPLVM': {}}

        for M in num_inducing_points:
            for latent_prior_coeff in latent_prior_coeffs:
                for max_temp in max_temps:
                    print('\t\t\t*** M = {}, latent_prior = {:.2e}, max_temp = {:.2f} ***'
                          .format(M, latent_prior_coeff, max_temp))
                    if (M, latent_prior_coeff, max_temp) in fitted_GPLVMs[rw]['rotated_GPLVM']:
                        continue

                    tf.reset_default_graph()

                    rotations_gplvm = RotationsGPLVM(
                        log_dir=save_dir + '_checkpoints',
                        W=W,
                        N=W.shape[0],
                        M=M,
                        num_orientations=8,
                        num_features=16,
                        latent_prior_coeff=latent_prior_coeff,
                        gamma=0.01,
                        beta_init=10000)

                    rotations_gplvm.fit(
                        W,
                        temperature_burnin=10000,
                        max_iter=100000,
                        eval_steps=100,
                        init_temperature=0,
                        max_temperature=max_temp,
                        init_lr=1e-2)

                    sess = rotations_gplvm.session
                    latent_space = sess.run(rotations_gplvm.t_gplvm_latent)
                    GPLVM_likelihood = sess.run(rotations_gplvm.t_gplvm_likelihood, rotations_gplvm.feed_dict)
                    beta = sess.run(rotations_gplvm.t_beta)

                    results = {'latent_space': latent_space,
                               'GPLVM_likelihood': GPLVM_likelihood,
                               'beta': beta}

                    fitted_GPLVMs[rw]['rotated_GPLVM'][(M, latent_prior_coeff, max_temp)] = results
                    pickle_file(save_file, fitted_GPLVMs)

                    print('')

class RotationsAlignment:
    def __init__(
        self,
        N, M, K, W,
        num_orientations,
        num_features,
        session=None,
        graph=None,
        log_dir='checkpoints_rotations_alignment',
        alignment_method='energy',
        rotations_method='basis',
        max_temperature=None,
        gplvm_latent_prior_coeff=0.1,
        gplvm_beta_init=1.0,
        gplvm_alpha=1.0,
        gplvm_gamma=0.01,
        optimised_temperature=False,
        rec_loss_coeff=1.0,
        means=None,
        covars=None,
        covar_init_std=1e-1,
        jitter=1e-6,
        dtype=tf.float64):

        self.N = N
        self.M = M
        self.K = K
        self.alignment_method = alignment_method
        self.rotations_method = rotations_method
        self.num_orientations = num_orientations
        self.num_features = num_features
        self.optimised_temperature = optimised_temperature
        self.jitter = jitter
        self.latent_prior_coeff = gplvm_latent_prior_coeff
        self.rec_loss_coeff = rec_loss_coeff
        self.dtype = dtype

        if means is not None:
            assert len(means) == self.K
        if covars is not None:
            assert len(covars) == self.K

        if graph is None:
            graph = tf.Graph()
        if session is None:
            session = tf.Session(graph=graph)

        self.graph = graph
        self.session = session
        self.log_dir = log_dir

        with self.graph.as_default():
            self.t_W = tf.constant(W, dtype=self.dtype)

            self.t_orientation_estimates = tf.Variable(1e-2 * np.random.randn(N, 1), dtype=self.dtype)
            self.t_orientation_estimates = 2 * np.pi * tf.nn.sigmoid(self.t_orientation_estimates)


            if not optimised_temperature:
                self.t_temperature = tf.placeholder(shape=(), dtype=self.dtype)
            elif max_temperature is None:
                self.t_temperature = tf.exp(tf.Variable(0, dtype=self.dtype))
            else:
                self.t_temperature = max_temperature * tf.nn.sigmoid(tf.Variable(0, dtype=self.dtype))


            if self.rotations_method == 'basis':
                self.t_shift_coeffs = \
                    rotations_linear_combination_von_mises(
                        self.t_orientation_estimates,
                        self.t_temperature,
                        self.num_orientations)

                self.t_shift_coeffs_zero = \
                    rotations_linear_combination_von_mises(
                        0 * self.t_orientation_estimates,
                        self.t_temperature,
                        self.num_orientations)

                basis = np.stack([[np.roll(d, shift=i, axis=-1) for i in range(num_orientations)] for d in W])
                self.t_basis = tf.constant(basis, dtype=self.dtype)

                self.t_shifted_W = tf.reshape(self.t_shift_coeffs, (N, num_orientations, 1, 1)) * self.t_basis
                self.t_shifted_W = tf.reduce_sum(self.t_shifted_W, axis=1)
                self.t_shifted_W = tf.reshape(self.t_shifted_W, (N, -1))

                self.t_shifted_W_zero = tf.reshape(self.t_shift_coeffs_zero, (N, num_orientations, 1, 1)) * self.t_basis
                self.t_shifted_W_zero = tf.reduce_sum(self.t_shifted_W_zero, axis=1)
            elif self.rotations_method == 'matrix':
                t_pi = tf.constant(np.pi, self.dtype)
                t_zero = tf.constant(0, self.dtype)
                t_delta = 2 * t_pi / num_orientations

                t_is = tf.tile(tf.constant(np.arange(num_orientations).reshape(1, -1), tf.int32), [num_orientations, 1])
                t_js = tf.tile(tf.constant(np.arange(num_orientations).reshape(-1, 1), tf.int32), [1, num_orientations])
                t_ks = tf.cast(tf.mod(t_js - t_is, num_orientations), self.dtype)

                def R(t_phi):
                    r = tf.maximum(
                        t_zero, tf.minimum(1.0 - (t_phi - t_ks * t_delta) / t_delta,
                                           (t_phi - (t_ks - 1) * t_delta) / t_delta))
                    return r

                self.rotation_matrices = tf.stack([R(self.t_orientation_estimates[n,0]) for n in range(N)])
                self.t_shifted_W = tf.einsum('noj,nfj->nfo', self.rotation_matrices, self.t_W)
                self.t_shifted_W = tf.reshape(self.t_shifted_W, (N, -1))

                self.t_shifted_W_zero = self.t_W
            else:
                raise Exception('Unknown rotations method')

            self.rec_loss = tf.reduce_sum(tf.square(self.t_shifted_W_zero - W), axis=(1, 2))
            self.rec_loss = tf.reduce_mean(self.rec_loss)

            if self.alignment_method == 'gplvm':
                self.t_gplvm_latent = tf.Variable(1e-2 * np.random.randn(N, 2), dtype=self.dtype)

                self.t_alpha = tf.constant(gplvm_alpha, dtype=tf.float64)
                self.t_gamma = tf.constant(gplvm_gamma, dtype=tf.float64)

                self.t_gplvm_kernel = create_squared_exp_kernel(self.t_alpha, self.t_gamma)

                self.t_beta = tf.exp(tf.Variable(np.log(gplvm_beta_init), dtype=self.dtype))
                self.t_Z = tf.Variable(1e-2 * np.random.randn(M, 2), dtype=self.dtype)
                self.t_gplvm_gp = create_sparse_gp(
                        self.t_gplvm_latent,
                        self.t_shifted_W,
                        self.t_beta,
                        self.t_gplvm_kernel,
                        self.t_Z,
                        t_hyper_prior=0)

                self.t_gplvm_likelihood = -self.t_gplvm_gp.t_neg_log_likelihood
                self.t_loss = \
                    self.latent_prior_coeff * tf.reduce_sum(self.t_gplvm_latent**2) - \
                    (1 / (N * num_features)) * self.t_gplvm_likelihood + \
                    self.rec_loss_coeff * self.rec_loss
            elif self.alignment_method == 'energy':
                self.t_loss = \
                    tf.reduce_mean(tf_pdist2(self.t_shifted_W, self.t_shifted_W)) + \
                    self.rec_loss_coeff * self.rec_loss
            elif self.alignment_method == 'gmm':
                self.t_gamma_raw = \
                    tf.Variable(
                        1e-2 * np.random.randn(N, K),
                        dtype=self.dtype)
                self.t_gamma = tf.nn.softmax(self.t_gamma_raw, axis=1)

                self.likelihoods = []
                for k in range(self.K):
                    if means is None:
                        t_mu_k = tf.Variable(
                            1e-2 * np.random.randn(self.num_features * self.num_orientations),
                            dtype=self.dtype)
                    else:
                        t_mu_k = tf.constant(
                            means[k],
                            dtype=self.dtype)

                    if covars is None:
                        t_sigma_k = tf.Variable(
                            covar_init_std * init_triangular(self.num_features * self.num_orientations),
                            dtype=self.dtype)
                        t_Sigma_k_sqrt = vec_to_tri(
                            t_sigma_k,
                            self.num_features * self.num_orientations)
                        t_Sigma_k = tf.matmul(
                            t_Sigma_k_sqrt,
                            t_Sigma_k_sqrt,
                            transpose_b=True) + \
                            1e-6 * tf.eye(self.num_features * self.num_orientations, dtype=self.dtype)
                    else:
                        t_Sigma_k = tf.constant(
                            covars[k],
                            dtype=self.dtype)

                    t_likelihood_dist_k = tfd.MultivariateNormalFullCovariance(
                            loc=t_mu_k,
                            covariance_matrix=t_Sigma_k)
                    self.likelihoods.append(t_likelihood_dist_k)

                self.t_gmm_likelihood = tf.stack(
                    [l.log_prob(self.t_shifted_W)
                     for k, l in enumerate(self.likelihoods)], axis=1)
                self.t_gmm_likelihood = tf.reduce_sum(
                    self.t_gamma * self.t_gmm_likelihood, axis=1)
                self.t_gmm_likelihood = tf.reduce_mean(self.t_gmm_likelihood)
                self.t_loss = -self.t_gmm_likelihood + \
                               self.rec_loss_coeff * self.rec_loss
            else:
                raise Exception('Unknown alignment method')

            self.t_alignment_loss = self.t_loss - \
                    self.rec_loss_coeff * self.rec_loss

    @property
    def saver(self):
        with self.graph.as_default():
            self._saver = tf.train.Saver(max_to_keep=1)
        return self._saver

    def save(self):
        with self.graph.as_default():
            self.saver.save(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def load(self):
        with self.graph.as_default():
            self.saver.restore(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def init_uninitialised_vars(self):
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.session.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        self.session.run(tf.variables_initializer(uninitialized_vars))

    def fit(
        self,
        W,
        init_temperature=0,
        max_temperature=3,
        temperature_burnin=5000,
        max_iter=10000,
        eval_steps=100,
        init_lr=1e-2,
        lr_decay_steps=5,
        patience=5,
        conn=None,
        verbose=True):

        with self.graph.as_default():
            t_lr = tf.placeholder(dtype=self.dtype)
            optimiser = tf.train.AdamOptimizer(
                learning_rate=t_lr).minimize(self.t_loss)
            self.init_uninitialised_vars()

            learning_rate = init_lr
            self.feed_dict = {}
            temperatures = np.linspace(
                init_temperature,
                max_temperature,
                temperature_burnin + 1)

            not_improved = 0
            iter_num = 0
            best_loss = np.inf
            for lr_decay_step in range(lr_decay_steps):
                self.feed_dict[t_lr] = learning_rate
                while iter_num < max_iter:
                    # training step
                    if not self.optimised_temperature:
                        self.feed_dict[self.t_temperature] = temperatures[min(iter_num, temperature_burnin)]
                    self.session.run(optimiser, self.feed_dict)
                    iter_num += 1

                    # validate/save periodically
                    if not (iter_num % eval_steps):
                        if conn is not None:
                            conn.connect()
                        loss = self.session.run(self.t_loss, self.feed_dict)
                        if verbose:
                            print('{:4d} | Loss: {:.2f}'.format(iter_num, loss))
                        if best_loss is np.inf or loss < best_loss - 1e-3 * np.abs(best_loss) or iter_num == temperature_burnin:
                            best_loss = loss
                            self.save()
                            not_improved = 0
                        else:
                            not_improved += int(1 * (iter_num > temperature_burnin))
                        if not_improved == patience:
                            self.load()
                            iter_num -= patience * eval_steps
                            not_improved = 0
                            break

                learning_rate /= np.sqrt(10)
                print('Reducing learning rate to {:f}'.format(learning_rate))

            if max_iter > eval_steps:
                self.load()
            if not self.optimised_temperature:
                self.feed_dict[self.t_temperature] = temperatures[min(iter_num, temperature_burnin)]

class RotationsGMM:
    def __init__(
        self,
        N, K, W,
        num_orientations,
        num_features,
        session=None,
        graph=None,
        log_dir='checkpoints_rotated_GMM',
        means=None,
        covars=None,
        alpha=100,
        jitter=1e-6,
        covar_init_std=1e1,
        optimised_temperature=False,
        dtype=tf.float64,
        gamma=None):

        self.N = N
        self.K = K
        self.num_orientations = num_orientations
        self.num_features = num_features
        self.optimised_temperature = optimised_temperature
        self.jitter = jitter
        self.dtype = dtype

        if means is not None:
            assert len(means) == self.K
        if covars is not None:
            assert len(covars) == self.K

        if graph is None:
            graph = tf.Graph()
        if session is None:
            session = tf.Session(graph=graph)

        self.graph = graph
        self.session = session
        self.log_dir = log_dir

        with self.graph.as_default():
            self.t_gamma_raw = tf.Variable(
                1e-2 * np.random.randn(N, K),
                dtype=self.dtype)
            if gamma is not None:
                self.t_gamma_raw = 100 * tf.one_hot(gamma, K, dtype=tf.float64)
            self.t_gamma = tf.nn.softmax(self.t_gamma_raw, axis=1)
            self.N_k = tf.reduce_sum(self.t_gamma, axis=0)

            self.t_alpha = tf.constant(alpha, dtype=self.dtype)

            self.t_orientation_estimates = tf.Variable(1e-2 * np.random.randn(N, 1), dtype=self.dtype)
            self.t_orientation_estimates = 2 * np.pi * tf.nn.sigmoid(self.t_orientation_estimates)

            if not optimised_temperature:
                self.t_temperature = tf.placeholder(shape=(), dtype=self.dtype)
            else:
                self.t_temperature = tf.exp(tf.Variable(0, dtype=self.dtype))

            self.t_shift_coeffs = \
                rotations_linear_combination_von_mises(
                    self.t_orientation_estimates,
                    self.t_temperature,
                    self.num_orientations)

            self.t_shift_coeffs_zero = \
                rotations_linear_combination_von_mises(
                    0 * self.t_orientation_estimates,
                    self.t_temperature,
                    self.num_orientations)

            basis = np.stack([[np.roll(d, shift=i, axis=-1) for i in range(num_orientations)] for d in W])
            self.t_basis = tf.constant(basis, dtype=self.dtype)

            self.t_shifted_W = tf.reshape(self.t_shift_coeffs, (N, num_orientations, 1, 1)) * self.t_basis
            self.t_shifted_W = tf.reduce_sum(self.t_shifted_W, axis=1)
            self.t_shifted_W = tf.reshape(self.t_shifted_W, (N, -1))

            self.t_shifted_W_zero = tf.reshape(self.t_shift_coeffs_zero, (N, num_orientations, 1, 1)) * self.t_basis
            self.t_shifted_W_zero = tf.reduce_sum(self.t_shifted_W_zero, axis=1)

            self.likelihoods = []
            for k in range(self.K):
                if means is None:
                    t_mu_k = tf.Variable(
                        1e-2 * np.random.randn(self.num_features * self.num_orientations),
                        dtype=self.dtype)
                    # t_mu_k = tf.Variable(
                    #     W.reshape(1, -1)[k],
                    #     dtype=self.dtype)
                else:
                    t_mu_k = tf.constant(
                        means[k],
                        dtype=self.dtype)

                if covars is None:
                    t_sigma_k = tf.Variable(
                        covar_init_std * init_triangular(self.num_features * self.num_orientations),
                        dtype=self.dtype)
                    t_Sigma_k_sqrt = vec_to_tri(
                        t_sigma_k,
                        self.num_features * self.num_orientations)
                    t_Sigma_k = tf.matmul(
                        t_Sigma_k_sqrt,
                        t_Sigma_k_sqrt,
                        transpose_b=True) + \
                        1e-6 * tf.eye(self.num_features * self.num_orientations, dtype=self.dtype)
                else:
                    t_Sigma_k = tf.constant(
                        covars[k],
                        dtype=self.dtype)

                t_likelihood_dist_k = tfd.MultivariateNormalFullCovariance(
                        loc=t_mu_k,
                        covariance_matrix=t_Sigma_k)
                self.likelihoods.append(t_likelihood_dist_k)

            self.t_E_pRZ = tf.stack(
                [l.log_prob(self.t_shifted_W)
                 for k, l in enumerate(self.likelihoods)], axis=1)
            self.t_E_pRZ = tf.reduce_sum(
                self.t_gamma * self.t_E_pRZ, name='E_pRZ')

            self.t_E_pZpi = tf.reduce_sum(
                self.t_gamma * tf.digamma(self.t_alpha + self.N_k), axis=1) - \
                tf.digamma(self.t_alpha * self.K + self.N)
            self.t_E_pZpi = tf.reduce_sum(self.t_E_pZpi, name='E_pZpi')

            self.t_E_ppi = (self.t_alpha - 1) * tf.reduce_sum(
                tf.digamma(self.t_alpha + self.N_k) - \
                tf.digamma(self.t_alpha * self.K + self.N)
                ) - \
                (self.K * tf.lgamma(self.t_alpha) - tf.lgamma(self.K * self.t_alpha))
            self.t_E_ppi = tf.identity(self.t_E_ppi, name='E_ppi')

            self.t_E_qZ = tf.reduce_sum(
                self.t_gamma * tf.log(self.t_gamma), name='E_qZ')

            self.t_E_qpi = tf.reduce_sum(
                (self.t_alpha - 1 + self.N_k) * tf.digamma(self.t_alpha + self.N_k)) - \
                ((self.t_alpha - 1) * self.K + self.N) * tf.digamma(self.t_alpha * self.K + self.N) - \
                (tf.reduce_sum(tf.lgamma(self.t_alpha + self.N_k)) - tf.lgamma(self.t_alpha * self.K + self.N))
            self.t_E_qpi = tf.identity(self.t_E_qpi, name='E_qpi')

            # self.t_lower_bound = \
            #     1 * self.t_E_pRZ + \
            #     0 * self.t_E_pZpi + \
            #     0 * self.t_E_ppi - \
            #     0 * self.t_E_qZ - \
            #     0 * self.t_E_qpi

            def tf_pdist2(X, Y):
                xx = tf.reduce_sum(X * X, axis=1, keepdims=True)
                yy = tf.reduce_sum(Y * Y, axis=1, keepdims=True)
                return xx + tf.transpose(yy) - 2.0 * tf.matmul(X, Y, transpose_b=True)

            self.rec_loss = tf.reduce_sum(tf.square(self.t_shifted_W_zero - W), axis=(1, 2))
            self.rec_loss = tf.reduce_mean(self.rec_loss)

            self.t_lower_bound = \
                    -tf.reduce_mean(tf_pdist2(self.t_shifted_W, self.t_shifted_W)) - \
                    0.25 * self.rec_loss
            self.t_neg_lower_bound = tf.identity(
                -self.t_lower_bound, name='neg_lower_bound')

    @property
    def saver(self):
        with self.graph.as_default():
            self._saver = tf.train.Saver(max_to_keep=1)
        return self._saver

    def save(self):
        with self.graph.as_default():
            self.saver.save(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def load(self):
        with self.graph.as_default():
            self.saver.restore(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def init_uninitialised_vars(self):
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.session.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        self.session.run(tf.variables_initializer(uninitialized_vars))

    def fit(
        self,
        W,
        init_temperature=0,
        max_temperature=3,
        temperature_burnin=5000,
        max_iter=10000,
        eval_steps=100,
        init_lr=1e-2,
        lr_decay_steps=5,
        patience=5,
        verbose=True):

        with self.graph.as_default():
            t_lr = tf.placeholder(dtype=self.dtype)
            optimiser = tf.train.AdamOptimizer(
                learning_rate=t_lr).minimize(self.t_neg_lower_bound)
            t_loss = self.t_neg_lower_bound
            self.init_uninitialised_vars()

            learning_rate = init_lr
            self.feed_dict = {} # {self.t_W: W}
            temperatures = np.linspace(
                init_temperature,
                max_temperature,
                temperature_burnin + 1)

            not_improved = 0
            iter_num = 0
            best_loss = np.inf
            for lr_decay_step in range(lr_decay_steps):
                self.feed_dict[t_lr] = learning_rate
                while iter_num < max_iter:
                    # training step
                    if not self.optimised_temperature:
                        self.feed_dict[self.t_temperature] = temperatures[min(iter_num, temperature_burnin)]
                    self.session.run(optimiser, self.feed_dict)
                    iter_num += 1

                    # validate/save periodically
                    if not (iter_num % eval_steps):
                        loss = self.session.run(t_loss, self.feed_dict)
                        if verbose:
                            print('{:4d} | Loss: {:.2f}'.format(iter_num, loss))
                        if best_loss is np.inf or loss < best_loss - 1e-3 * np.abs(best_loss) or iter_num == temperature_burnin:
                            best_loss = loss
                            self.save()
                            not_improved = 0
                        else:
                            not_improved += int(1 * (iter_num > temperature_burnin))
                        if not_improved == patience:
                            self.load()
                            iter_num -= patience * eval_steps
                            not_improved = 0
                            break

                learning_rate /= np.sqrt(10)
                print('Reducing learning rate to {:f}'.format(learning_rate))

            if max_iter > eval_steps:
                self.load()
            if not self.optimised_temperature:
                self.feed_dict[self.t_temperature] = temperatures[min(iter_num, temperature_burnin)]

    def get_parameters(self, prefix=''):
        sess = self.session

        params = {
            prefix + '_score': sess.run(self.t_E_pRZ, self.feed_dict),
            prefix + '_reg': sess.run(self.t_orientations_reg, self.feed_dict),
            prefix + '_lower_bound': sess.run(self.t_neg_lower_bound, self.feed_dict),
            prefix + '_lower_bound_unreg': sess.run(self.t_neg_lower_bound_unreg, self.feed_dict),
            prefix + '_gamma': sess.run(self.t_gamma),
            prefix + '_pred': np.argmax(sess.run(self.t_gamma), axis=1),
            prefix + '_W_oriented': sess.run(self.t_W_oriented, self.feed_dict),
            prefix + '_orientations_neurons': sess.run(self.t_orientations_neurons),
            prefix + '_orientations_cluster_features': sess.run(self.t_orientations_cluster_features),
            prefix + '_means': [sess.run(l.mean()) for l in self.likelihoods],
            prefix + '_covars': [sess.run(l.covariance()) for l in self.likelihoods],
            prefix + '_orientations_temperature': sess.run(self.t_orientations_temperature)
        }
        return params

class OrientationsGMM:
    def __init__(
        self,
        N, K,
        num_orientations,
        num_features,
        mode,
        session=None,
        graph=None,
        log_dir='checkpoints_oriented_GMM',
        means=None,
        covars=None,
        alpha=1.0,
        jitter=1e-6,
        orientations_reg_coeff=1,
        covar_init_std=1e1,
        dtype=tf.float64,
        gamma=None):

        self.N = N
        self.K = K
        self.mode = mode
        self.num_orientations = num_orientations
        self.num_features = num_features
        self.jitter = jitter
        self.orientations_reg_coeff = orientations_reg_coeff
        self.dtype = dtype

        if means is not None:
            assert len(means) == self.K
        if covars is not None:
            assert len(covars) == self.K

        if graph is None:
            graph = tf.Graph()
        if session is None:
            session = tf.Session(graph=graph)

        self.graph = graph
        self.session = session
        self.log_dir = log_dir

        self.features_vars = []
        self.mixture_vars = []

        with self.graph.as_default():
            self.t_W = tf.placeholder(
                shape=(self.N, self.num_orientations * self.num_features),
                dtype=self.dtype)

            # self.t_orientations_temperature = tf.placeholder(
            #     shape=(), dtype=self.dtype)
            self.t_orientations_temperature = \
                    tf.exp(tf.Variable(np.log(0.5), dtype=self.dtype))

            self.t_alpha = tf.constant(alpha, dtype=self.dtype)

            self.t_gamma_raw = tf.Variable(
                1e-2 * np.random.randn(N, K),
                dtype=self.dtype)
            self.mixture_vars.append(self.t_gamma_raw)
            if gamma is not None:
                self.t_gamma_raw = 100 * tf.one_hot(gamma, K, dtype=tf.float64)
            self.t_gamma = tf.nn.softmax(self.t_gamma_raw, axis=1)
            self.N_k = tf.reduce_sum(self.t_gamma, axis=0)

            if mode == 'combination_of_orientations':
                self.t_gamma_sign_raw = tf.Variable(
                    1e-2 * np.random.randn(N, K, num_features),
                    dtype=self.dtype)
                self.t_gamma_sign = tf.nn.tanh(self.t_gamma_sign_raw)
                self.t_gamma_sign_split = tf.unstack(self.t_gamma_sign, axis=-1)

            self.t_orientations_neurons = tf.Variable(
                1e-2 * np.random.randn(N, K, 1),
                dtype=self.dtype)
            self.features_vars.append(self.t_orientations_neurons)
            self.t_orientations_neurons = \
                2 * np.pi * tf.nn.sigmoid(self.t_orientations_neurons)

            self.t_orientations_cluster_features = tf.Variable(
                1e-2 * np.random.randn(1, K, num_features - 1),
                dtype=self.dtype)
            self.features_vars.append(self.t_orientations_cluster_features)
            self.t_orientations_cluster_features = \
                2 * np.pi * tf.nn.sigmoid(self.t_orientations_cluster_features)
            self.t_orientations_cluster_features = tf.concat(
                [tf.zeros((1, K, 1), dtype=self.dtype), self.t_orientations_cluster_features],
                axis=-1)

            self.t_orientations = \
                self.t_orientations_neurons + \
                self.t_orientations_cluster_features

            self.t_orientations_zero = \
                0 * self.t_orientations_neurons + \
                self.t_orientations_cluster_features

            self.t_orientations_shifts = \
                orientations_linear_combination_von_mises(
                    self.t_orientations,
                    self.t_orientations_temperature,
                    self.num_orientations)

            self.t_orientations_shifts_zero = \
                orientations_linear_combination_von_mises(
                    self.t_orientations_zero,
                    self.t_orientations_temperature,
                    self.num_orientations)

            if means is None:
                self.t_W_oriented_cluster_features = tf.Variable(
                    1e-2 * np.random.randn(K, num_features),
                    dtype=self.dtype)
                # if mode == 'combination_of_orientations':
                #     self.t_W_oriented_cluster_features = \
                #             tf.exp(self.t_W_oriented_cluster_features)
            else:
                self.t_W_oriented_cluster_features = \
                        tf.stack([tf.constant(m, dtype=self.dtype) for m in means], axis=0)

            t_W_oriented_feature_split = \
                    tf.unstack(self.t_W_oriented_cluster_features, axis=-1)

            if mode == 'single_orientation':
                t_orientations_shifts_split = \
                        tf.unstack(self.t_orientations_shifts, axis=-1)
                t_orientations_shifts_zero_split = \
                        tf.unstack(self.t_orientations_shifts_zero, axis=-1)
                self.t_cluster_feature_combination_coeffs = \
                        tf.zeros((), dtype=self.dtype)
            elif mode == 'combination_of_orientations':
                self.t_cluster_feature_combination_coeffs = \
                        tf.Variable(1e-2 * np.random.randn(num_orientations, K, num_features),
                                    dtype=self.dtype)
                self.t_cluster_feature_combination_coeffs_norm = \
                        tf.sqrt(
                            tf.reduce_sum(
                                self.t_cluster_feature_combination_coeffs**2,
                                axis=0,
                                keepdims=True)) + jitter
                self.t_cluster_feature_combination_coeffs = \
                        self.t_cluster_feature_combination_coeffs / self.t_cluster_feature_combination_coeffs_norm
                self.t_basis = tf.stack(
                        [tf.roll(self.t_cluster_feature_combination_coeffs, shift=i, axis=0)
                         for i in range(num_orientations)])
                self.t_orientations_shifts = \
                        tf.reduce_sum(
                            tf.expand_dims(self.t_orientations_shifts, axis=1) * \
                            tf.expand_dims(self.t_basis, axis=2),
                            axis=0)
                self.t_orientations_shifts_zero = \
                        tf.reduce_sum(
                            tf.expand_dims(self.t_orientations_shifts_zero, axis=1) * \
                            tf.expand_dims(self.t_basis, axis=2),
                            axis=0)
                t_orientations_shifts_split = \
                        tf.unstack(self.t_orientations_shifts, axis=-1)
                t_orientations_shifts_zero_split = \
                        tf.unstack(self.t_orientations_shifts_zero, axis=-1)
            else:
                raise Exception('Unknown orientations mode: {}'.format(mode))

            t_feature_diffs = []
            t_feature_diffs_sign = []
            t_W_oriented = []
            t_W_oriented_zero = []
            for feature in range(self.num_features):
                t_W_feature = tf.transpose(self.t_W[:, feature::num_features])
                t_W_feature = tf.expand_dims(t_W_feature, -1)

                t_feature_diff = tf.square(
                    t_W_oriented_feature_split[feature] * \
                    t_orientations_shifts_split[feature] - t_W_feature)
                t_feature_diffs.append(t_feature_diff)
                # t_W_oriented.append(
                #         tf.reduce_sum(
                #             t_orientations_shifts_split[feature] * t_W_feature,
                #             axis=0))
                if mode == 'single_orientation':
                    t_W_oriented.append(
                            tf.reduce_sum(
                                tf.ones_like(t_orientations_shifts_split[feature]) * t_W_feature,
                                axis=0))

                    feature_weight_zero = \
                            tf.reduce_sum(
                                tf.ones_like(t_orientations_shifts_zero_split[feature]) * t_W_feature,
                                axis=0)
                    t_W_oriented_zero.append(t_orientations_shifts_zero_split[feature] * feature_weight_zero)
                    # t_W_oriented.append(tf.reduce_sum(t_W_feature, axis=0))
                elif mode == 'combination_of_orientations':
                    t_feature_norm_weight = \
                            tf.sqrt(tf.reduce_sum(tf.ones_like(t_orientations_shifts_split[feature]) * t_W_feature**2, axis=0))
                    t_feature_norm_weight_zero = \
                            tf.sqrt(tf.reduce_sum(tf.ones_like(t_orientations_shifts_zero_split[feature]) * t_W_feature**2, axis=0))
                    t_gamma_sign_feature = self.t_gamma_sign_split[feature]
                    t_fearure_norm_weight_sign = \
                            t_feature_norm_weight * t_gamma_sign_feature
                    t_fearure_norm_weight_zero_sign = \
                            t_feature_norm_weight_zero * t_gamma_sign_feature
                    t_W_oriented.append(t_fearure_norm_weight_sign)
                    t_W_oriented_zero.append(t_orientations_shifts_zero_split[feature] * t_fearure_norm_weight_zero_sign)

                    t_feature_diff_sign = tf.square(
                        t_fearure_norm_weight_sign * \
                        tf.stop_gradient(t_orientations_shifts_split[feature]) - t_W_feature)
                    t_feature_diffs_sign.append(t_feature_diff_sign)
                else:
                    raise Exception('Unknown orientations mode: {}'.format(mode))

            self.t_W_oriented = tf.transpose(tf.stack(t_W_oriented, axis=-1), (1, 0, 2))

            self.t_W_oriented_zero = tf.reshape(tf.stack(t_W_oriented_zero, axis=0),
                                               (num_features * num_orientations, N, K))
            self.t_W_oriented_zero = tf.transpose(self.t_W_oriented_zero, (2, 1, 0))

            self.t_feature_diffs = t_feature_diffs
            self.t_feature_diffs_sign = t_feature_diffs_sign

            self.likelihoods = []
            for k in range(self.K):
                if means is None:
                    # t_mu_k = tf.Variable(
                    #     1e-2 * np.random.randn(self.num_features),
                    #     dtype=self.dtype)
                    t_mu_k = self.t_W_oriented_cluster_features[k]
                    self.mixture_vars.append(t_mu_k)
                else:
                    t_mu_k = tf.constant(
                        means[k],
                        dtype=self.dtype)

                if covars is None:
                    t_sigma_k = tf.Variable(
                        covar_init_std * init_triangular(num_features),
                        dtype=self.dtype)
                    self.mixture_vars.append(t_sigma_k)
                    t_Sigma_k_sqrt = vec_to_tri(
                        t_sigma_k,
                        self.num_features)
                    t_Sigma_k = tf.matmul(
                        t_Sigma_k_sqrt,
                        t_Sigma_k_sqrt,
                        transpose_b=True) + \
                        1e-6 * tf.eye(self.num_features, dtype=self.dtype)
                else:
                    t_Sigma_k = tf.constant(
                        covars[k],
                        dtype=self.dtype)

                t_likelihood_dist_k = tfd.MultivariateNormalFullCovariance(
                        loc=t_mu_k,
                        covariance_matrix=t_Sigma_k)
                self.likelihoods.append(t_likelihood_dist_k)

            self.t_E_pRZ = tf.stack(
                [l.log_prob(tf.stop_gradient(self.t_W_oriented[k]))
                 for k, l in enumerate(self.likelihoods)], axis=-1)
            self.t_E_pRZ = tf.reduce_sum(
                self.t_gamma * self.t_E_pRZ, name='E_pRZ')

            self.t_feature_diffs_stacked = tf.stack(self.t_feature_diffs, axis=0)
            self.t_feature_diffs = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.reduce_sum(self.t_gamma * self.t_feature_diffs_stacked, axis=-1),
                        axis=(0,1)))
            # self.t_feature_diffs = tf.reduce_mean(
            #         tf.reduce_sum(
            #             self.t_feature_diffs_stacked, axis=(0,1)))
            self.t_orientations_reg = tf.reduce_sum(self.t_feature_diffs)

            if mode == 'combination_of_orientations':
                self.t_feature_diffs_sign_stacked = tf.stack(self.t_feature_diffs_sign, axis=0)
                self.t_feature_diffs_sign = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.reduce_sum(self.t_gamma * self.t_feature_diffs_sign_stacked, axis=-1),
                            axis=(0,1)))
                self.t_sign_reg = tf.reduce_sum(self.t_feature_diffs_sign)
            else:
                self.t_sign_reg = tf.constant(0, dtype=self.dtype)

            self.t_E_pZpi = tf.reduce_sum(
                self.t_gamma * tf.digamma(self.t_alpha + self.N_k), axis=1) - \
                tf.digamma(self.t_alpha * self.K + self.N)
            self.t_E_pZpi = tf.reduce_sum(self.t_E_pZpi, name='E_pZpi')

            self.t_E_ppi = (self.t_alpha - 1) * tf.reduce_sum(
                tf.digamma(self.t_alpha + self.N_k) - \
                tf.digamma(self.t_alpha * self.K + self.N)
                ) - \
                (self.K * tf.lgamma(self.t_alpha) - tf.lgamma(self.K * self.t_alpha))
            self.t_E_ppi = tf.identity(self.t_E_ppi, name='E_ppi')

            self.t_E_qZ = tf.reduce_sum(
                self.t_gamma * tf.log(self.t_gamma), name='E_qZ')

            self.t_E_qpi = tf.reduce_sum(
                (self.t_alpha - 1 + self.N_k) * tf.digamma(self.t_alpha + self.N_k)) - \
                ((self.t_alpha - 1) * self.K + self.N) * tf.digamma(self.t_alpha * self.K + self.N) - \
                (tf.reduce_sum(tf.lgamma(self.t_alpha + self.N_k)) - tf.lgamma(self.t_alpha * self.K + self.N))
            self.t_E_qpi = tf.identity(self.t_E_qpi, name='E_qpi')

            self.t_lower_bound = \
                1 * self.t_E_pRZ - \
                self.orientations_reg_coeff * self.t_orientations_reg - \
                self.orientations_reg_coeff * self.t_sign_reg + \
                0 * self.t_E_pZpi + \
                0 * self.t_E_ppi - \
                0 * self.t_E_qZ - \
                0 * self.t_E_qpi
            self.t_reg_only = \
                self.orientations_reg_coeff * self.t_orientations_reg + \
                self.orientations_reg_coeff * self.t_sign_reg
            self.t_neg_lower_bound = tf.identity(
                -self.t_lower_bound, name='neg_lower_bound')

            self.t_neg_lower_bound_unreg = \
                self.t_neg_lower_bound - \
                self.orientations_reg_coeff * self.t_orientations_reg - \
                self.orientations_reg_coeff * self.t_sign_reg

    @property
    def saver(self):
        with self.graph.as_default():
            self._saver = tf.train.Saver(max_to_keep=1)
        return self._saver

    def save(self):
        with self.graph.as_default():
            self.saver.save(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def load(self):
        with self.graph.as_default():
            self.saver.restore(self.session, os.path.join(self.log_dir, 'model.ckpt'))

    def init_uninitialised_vars(self):
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.session.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        self.session.run(tf.variables_initializer(uninitialized_vars))

    def fit(
        self,
        W,
        max_orientations_temperature=1,
        init_orientations_temperature=0.5,
        max_iter=10000,
        eval_steps=100,
        init_lr=1e-2,
        lr_decay_steps=5,
        patience=5,
        reg_only=False,
        mixture_only=False,
        verbose=True):

        with self.graph.as_default():
            t_lr = tf.placeholder(dtype=self.dtype)
            if reg_only:
                optimiser = tf.train.AdamOptimizer(
                    learning_rate=t_lr).minimize(self.t_reg_only, var_list=self.features_vars)
                t_loss = self.t_reg_only
            elif mixture_only:
                optimiser = tf.train.AdamOptimizer(
                    learning_rate=t_lr).minimize(self.t_neg_lower_bound, var_list=self.mixture_vars)
                t_loss = self.t_neg_lower_bound
            else:
                optimiser = tf.train.AdamOptimizer(
                    learning_rate=t_lr).minimize(self.t_neg_lower_bound)
                t_loss = self.t_neg_lower_bound

            self.init_uninitialised_vars()

            learning_rate = init_lr
            self.feed_dict = {self.t_W: W}
            orientation_temperatures = np.linspace(
                init_orientations_temperature,
                max_orientations_temperature,
                max_iter + 1)

            not_improved = 0
            iter_num = 0
            best_loss = np.inf
            for lr_decay_step in range(lr_decay_steps):
                self.feed_dict[t_lr] = learning_rate
                while iter_num < max_iter:
                    # training step
                    # self.feed_dict[self.t_orientations_temperature] = orientation_temperatures[iter_num]
                    self.session.run(optimiser, self.feed_dict)
                    iter_num += 1

                    # validate/save periodically
                    if not (iter_num % eval_steps):
                        loss = self.session.run(t_loss, self.feed_dict)
                        if verbose:
                            print('{:4d} | Loss: {:.2e}'.format(iter_num, loss))
                        if best_loss is np.inf or loss < best_loss - 1e-3 * np.abs(best_loss):
                            best_loss = loss
                            self.save()
                            not_improved = 0
                        else:
                            not_improved += 1
                        if not_improved == patience:
                            self.load()
                            iter_num -= patience * eval_steps
                            not_improved = 0
                            break

                learning_rate /= np.sqrt(10)
                print('Reducing learning rate to {:f}'.format(learning_rate))

            if max_iter > eval_steps:
                self.load()
            # self.feed_dict[self.t_orientations_temperature] = orientation_temperatures[iter_num]

    def fit_separately(
        self,
        W,
        max_orientations_temperature=1,
        init_orientations_temperature=0.5,
        max_iter=10000,
        eval_steps=100,
        init_lr=1e-2,
        lr_decay_steps=5,
        patience=5,
        verbose=True):

        print('Fitting the orientations...')
        self.fit(
            W=W,
            max_orientations_temperature=max_orientations_temperature,
            init_orientations_temperature=init_orientations_temperature,
            max_iter=max_iter,
            eval_steps=eval_steps,
            init_lr=init_lr,
            lr_decay_steps=lr_decay_steps,
            patience=patience,
            reg_only=True,
            verbose=verbose)

        orientations_temperature = self.feed_dict[self.t_orientations_temperature]
        print('')
        print('Fitting the mixture model...')
        self.fit(
            W=W,
            max_orientations_temperature=orientations_temperature,
            init_orientations_temperature=orientations_temperature,
            max_iter=max_iter,
            eval_steps=eval_steps,
            init_lr=init_lr,
            lr_decay_steps=lr_decay_steps,
            patience=patience,
            reg_only=False,
            verbose=verbose)


    def compute_likelihood_bounds(self):
        return (self.session.run(self.t_neg_lower_bound, self.feed_dict),
                self.session.run(self.t_neg_lower_bound_unreg, self.feed_dict))

    def plot_oriented_space(self, true_assignments=None):
        oriented_W = self.session.run(self.t_W_oriented, self.feed_dict)
        oriented_W = np.transpose(oriented_W, (1, 0, 2))
        pred_assignments = np.argmax(self.session.run(self.t_gamma), axis=1)

        plt.figure(figsize=(12, 5))

        if true_assignments is not None:
            plt.subplot(1, 2, 1)
            for c in np.unique(true_assignments):
                plt.scatter(oriented_W[true_assignments==c, c, 0], oriented_W[true_assignments==c, c, 1], alpha=0.5)
            plt.title('Ground truth')

        plt.subplot(1, 2, 2)
        for c in np.unique(pred_assignments):
            plt.scatter(oriented_W[pred_assignments==c, c, 0], oriented_W[pred_assignments==c, c, 1], alpha=0.5)
        plt.title('Predicted clustering')

    def get_parameters(self, prefix=''):
        sess = self.session

        params = {
            prefix + '_score': sess.run(self.t_E_pRZ, self.feed_dict),
            prefix + '_rec_loss': sess.run(self.t_orientations_reg, self.feed_dict),
            prefix + '_sign_loss': sess.run(self.t_sign_reg, self.feed_dict),
            prefix + '_gamma': sess.run(self.t_gamma),
            prefix + '_pred': np.argmax(sess.run(self.t_gamma), axis=1),
            prefix + '_w_oriented': sess.run(self.t_W_oriented, self.feed_dict),
            prefix + '_w_oriented_zero': sess.run(self.t_W_oriented_zero, self.feed_dict),
            prefix + '_orientations_neurons': sess.run(self.t_orientations_neurons),
            prefix + '_orientations_cluster_features': sess.run(self.t_orientations_cluster_features),
            prefix + '_means': [sess.run(l.mean()) for l in self.likelihoods],
            prefix + '_covars': [sess.run(l.covariance()) for l in self.likelihoods],
            prefix + '_orientations_temperature': sess.run(self.t_orientations_temperature),
            prefix + '_cluster_feature_combination_coeffs': sess.run(self.t_cluster_feature_combination_coeffs)
        }
        return params

def orientations_gmm(t_W,
                     K,
                     num_orientations=8,
                     num_features=16,
                     t_orientations_temperature=2.0,
                     alpha=1.0,
                     jitter=1e-6):
    N, D = t_W.get_shape().as_list()
    assert D == num_orientations * num_features
    t_alpha = tf.constant(alpha, dtype=tf.float64)

    t_gamma = tf.Variable(1e-2 * np.random.randn(N, K))
    t_gamma = tf.nn.softmax(t_gamma, axis=1)
    N_k = tf.reduce_sum(t_gamma, axis=0)

    t_orientations = tf.Variable(1e-2 * np.random.randn(N, 1))
    t_orientations = 2 * np.pi * tf.nn.sigmoid(t_orientations)

    t_orientations_shifts = orientations_linear_combination_von_mises(
            t_orientations,
            t_orientations_temperature,
            num_orientations)

    t_orientations_shifts = tf.squeeze(t_orientations_shifts)

    oriented_W = []
    t_feature_diffs = []
    for feature in range(num_features):
        t_W_feature = tf.transpose(t_W[:, feature::num_features])
        t_W_oriented_feature = tf.Variable(np.random.randn(N)) + jitter * tf.random_normal(shape=(N,), dtype=tf.float64)
        t_feature_diff = tf.square(t_W_oriented_feature * t_orientations_shifts - t_W_feature)
        oriented_W.append(t_W_oriented_feature)
        t_feature_diffs.append(t_feature_diff)

    t_oriented_W = tf.stack(oriented_W, axis=-1)

    likelihoods = []
    for k in range(K):
        t_mu_k = tf.Variable(np.random.randn(num_features))

        t_sigma_k = tf.Variable(init_triangular(num_features))
        t_Sigma_k_sqrt = vec_to_tri(t_sigma_k, num_features)
        t_Sigma_k = tf.matmul(t_Sigma_k_sqrt, t_Sigma_k_sqrt, transpose_b=True) + jitter * tf.eye(num_features, dtype=tf.float64)

        t_likelihood_dist_k = tfd.MultivariateNormalFullCovariance(
                loc=t_mu_k,
                covariance_matrix=t_Sigma_k)
        likelihoods.append(t_likelihood_dist_k)

    t_E_pRZ = tf.stack([l.log_prob(t_oriented_W) for k, l in enumerate(likelihoods)], axis=-1)
    t_E_pRZ = tf.reduce_sum(t_gamma * t_E_pRZ, name='E_pRZ')
    t_E_pRZ = t_E_pRZ - tf.reduce_sum(tf.square(t_feature_diffs))

    t_E_pZpi = tf.reduce_sum(t_gamma * tf.digamma(t_alpha + N_k), axis=1) - tf.digamma(t_alpha * K + N)
    t_E_pZpi = tf.reduce_sum(t_E_pZpi, name='E_pZpi')

    t_E_ppi = (t_alpha - 1) * tf.reduce_sum(tf.digamma(t_alpha + N_k) - tf.digamma(t_alpha * K + N)) - \
              (K * tf.lgamma(t_alpha) - tf.lgamma(K * t_alpha))
    t_E_ppi = tf.identity(t_E_ppi, name='E_ppi')

    t_E_qZ = tf.reduce_sum(t_gamma * tf.log(t_gamma), name='E_qZ')

    t_E_qpi = tf.reduce_sum((t_alpha - 1 + N_k) * tf.digamma(t_alpha + N_k)) - ((t_alpha - 1) * K + N) * tf.digamma(t_alpha * K + N) - \
              (tf.reduce_sum(tf.lgamma(t_alpha + N_k)) - tf.lgamma(t_alpha * K + N))
    t_E_qpi = tf.identity(t_E_qpi, name='E_qpi')

    t_lower_bound = t_E_pRZ + t_E_pZpi + t_E_ppi - t_E_qZ - t_E_qpi
    t_neg_lower_bound = tf.identity(-t_lower_bound, name='neg_lower_bound')
    return t_neg_lower_bound, t_gamma, t_orientations_shifts, t_oriented_W, likelihoods


def orientations_linear_combination_von_mises(t_orientations,
                                              t_orientations_temperature,
                                              num_orientations):
    t_base_orientations = tf.constant(
            np.arange(0, 2 * np.pi, 2 * np.pi / num_orientations),
            dtype=tf.float64)
    t_base_orientations = tf.reshape(t_base_orientations, (num_orientations, 1, 1, 1))
    t_coefficients = tf.exp(t_orientations_temperature * tf.cos(t_base_orientations - t_orientations))
    t_coefficients = t_coefficients / tf.reduce_sum(t_coefficients, axis=0, keepdims=True)
    return t_coefficients

def rotations_linear_combination_von_mises(t_orientations,
                                           t_orientations_temperature,
                                           num_orientations):
    t_base_orientations = tf.constant(
            np.arange(0, 2 * np.pi, 2 * np.pi / num_orientations),
            dtype=tf.float64)
    t_base_orientations = tf.reshape(t_base_orientations, (1, num_orientations))
    t_coefficients = tf.exp(t_orientations_temperature * tf.cos(t_base_orientations - t_orientations))
    t_coefficients = t_coefficients / tf.reduce_sum(t_coefficients, axis=1, keepdims=True)
    return t_coefficients

def triangle(t_orientations, t_orientations_temperature, num_orientations, eps=1e-3):
    t_base_orientations = tf.constant(
            np.arange(0, 2 * np.pi, 2 * np.pi / num_orientations),
            dtype=tf.float64)
    t_base_orientations = tf.reshape(t_base_orientations, (num_orientations, 1, 1, 1))
    t_width = tf.constant(2 * np.pi / num_orientations, dtype=tf.float64)
    t_threshold = tf.cos(t_width) + eps
    t_cos_diff = tf.cos(t_orientations - t_base_orientations)
    idx = tf.nn.relu(tf.sign(t_cos_diff - t_threshold))
    weights = idx * (t_cos_diff - t_threshold)
    return weights / tf.reduce_sum(weights, axis=0, keepdims=True) + (1 / t_orientations_temperature)

def vec_to_tri(tri, N):
    """ map from vector to lower triangular matrix (adapted from gpflow) """
    indices = list(zip(*np.tril_indices(N)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int32)
    tri_part = tf.scatter_nd(indices=indices, shape=[N, N], updates=tri)
    return tri_part

def init_triangular(N, diag=None):
    """ Initialize lower triangular parametrization for  covariance matrices (adapted from gpflow) """
    I = int(N*(N+1)/2)
    indices = list(zip(*np.tril_indices(N)))
    diag_indices = np.array([idx for idx, (x, y) in enumerate(indices) if x == y])
    I = np.zeros(I)
    if diag is None:
        I[diag_indices] = 1
    else:
        I[diag_indices] = diag
    return I
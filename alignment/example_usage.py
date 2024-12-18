import os
import numpy as np
from mixture_trained_weights import RotationsAlignment, normalise_W
import tensorflow
import tensorflow_probability
import scipy

path = 'example.npy' # a file of shape N x dim, where N is number fo neurons and dim is dimensionality
save_path = 'aligned_example.npy'

def align(W, num_features=16, num_orientations=8):
    # W = angles[dim][s]
    W = normalise_W(W, normalise=True)
    W = np.transpose(np.reshape(W, (W.shape[0], num_orientations, num_features)), (0, 2, 1))

    rotations_alignment = RotationsAlignment(
            W=W,
            N=W.shape[0],
            M=1,
            K=1,
            max_temperature=1.5,
            log_dir=f'alignment_checkpoints',
            num_orientations=num_orientations,
            num_features=num_features,
            alignment_method='energy',
            rotations_method='basis',
            rec_loss_coeff=1.5,
            gplvm_latent_prior_coeff=1,
            optimised_temperature=True
    )

    rotations_alignment.fit(
        W,
        temperature_burnin=0,
        max_iter=100_000, 
        eval_steps=25,
        init_temperature=0.5,
        max_temperature=1,
        init_lr=0.01,
        conn=None
        )

    sess = rotations_alignment.session
    feed_dict = rotations_alignment.feed_dict
    aligned_readouts = sess.run(rotations_alignment.t_shifted_W, feed_dict)

    return aligned_readouts

if __name__ == "__main__":
    W = np.load(path)
    aligned_readouts = align(W, num_features=16, num_orientations=8)
    np.save(save_path, aligned_readouts)



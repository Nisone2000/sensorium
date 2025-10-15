import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from PIL import Image  # For GIF creation
import os
import warnings

import numpy as np
import pandas as pd
import skimage
import torch

import neuralpredictors

warnings.filterwarnings("ignore")

from nnfabrik.builder import get_data, get_model
from nnfabrik.utility.nn_helpers import set_random_seed
import studenttmixture
from studenttmixture.em_student_mixture import EMStudentMixture
from sklearn.mixture import GaussianMixture

seed = 12345
set_random_seed(seed)
torch.cuda.is_available()
cuda_number = 3
n_components=5
torch.cuda.set_device(f"cuda:{cuda_number}")
clusters = [5,10,20]

weights = torch.load(f'path/final_seed_{seed}.pth')
features_list = []
for key, tensor in weights.items():
    if key.endswith('._features') and key.startswith('readout.'):
        features = tensor.detach().squeeze().T.cpu().numpy()
        features_list.append(features)
features = np.vstack(features_list)


for n_components in clusters:
  mixture_model = GaussianMixture(n_components=n_components, 
                                  covariance_type='diag', 
                                  tol=0.001, 
                                  reg_covar=1e-06, 
                                  max_iter=100, 
                                  n_init=50, 
                                  init_params='kmeans',
                                  random_state=seed,
                                  verbose=1, 
                                  verbose_interval=10
                                  )

  gn = mixture_model.fit(features)
  predictions = gn.predict(features)
  np.save(f"predictions/predictions_GMM_without_KL{n_components}_cluster_seed_{seed}_GMM.npy", predictions)
  print(f'Cluster {n_components} complete' )
import neuralpredictors
import neuralpredictors

import torch
import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from nnfabrik.builder import get_data, get_model, get_trainer
import skimage
import os

torch.cuda.set_device("cuda:6")

pre = "/usr/users/agecker/datasets/sensorium_2022_pictures/real_dataset/"
print(pre)
filenames = [f"{pre}{i}/" for i in os.listdir(pre)]
print(filenames)

dataset_fn = 'sensorium.datasets.static_loaders'
dataset_config = {'paths': filenames,
                 'normalize': True,
                 'include_behavior': True,
                 'include_eye_position': True,
                 'batch_size': 128,
                 'scale':.25,
                 }

dataloaders = get_data(dataset_fn, dataset_config)

model_fn = 'sensorium.models.stacked_core_full_gauss_readout'
model_config = {'pad_input': False,  #no padding
              'stack': -1,  #no stacking
              'layers': 3,
              'input_kern': 9,  #filter dimension 9x9
              'gamma_input': 6.3831,
              'gamma_readout': 0.0076,
              'hidden_kern': 7,  #convolution in hidden layers
              'hidden_channels': 128, #hidden filters that are applied
              'grid_mean_predictor': {'type': 'cortex', #submodel, uses positions of the brain cells to help to learn their receptive fields (because often cells that are actually close in the brain look in close pixels areas)
              'input_dimensions': 2,
              'hidden_layers': 1,
              'hidden_features': 30,
              'final_tanh': True}, #applying tanh activation function in final layer
              'depth_separable': True, #depthwise separable convolutions
              'init_sigma': 0.1, #nitial standard deviation for the weights
              'init_mu_range': 0.3, #range for initial mean of weights
              'gauss_type': 'full', #Gaussian type, full covariance matrix
              'linear': True, #include linear layers
              'shifter': True, #include shifting?
               }
model = get_model(model_fn=model_fn,
                  model_config=model_config,
                  dataloaders=dataloaders,
                  seed=42,)

trainer_fn = "sensorium.training.standard_trainer"

trainer_config = {'max_iter': 200,
                 'verbose': False,
                 'lr_decay_steps': 4,
                 'avg_loss': False,
                 'lr_init': 0.009,
                 'device' : f"cuda:6",
                 'wandb_model_congfig ': 'sensorium_2022_pictures',
                 'wandb_dataset_config' : 'CNN',
                 'wandb_name' : 'LN_test1'
                 }

trainer = get_trainer(trainer_fn=trainer_fn, 
                     trainer_config=trainer_config)     

validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)                              
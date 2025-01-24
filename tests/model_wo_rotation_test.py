import os
import warnings

import numpy as np
import pandas as pd
import skimage
import torch
import argparse

import neuralpredictors

# import matplotlib.pyplot as plt
# import seaborn as sns


warnings.filterwarnings("ignore")

from nnfabrik.builder import get_data, get_model, get_trainer
from nnfabrik.utility.nn_helpers import set_random_seed


parser = argparse.ArgumentParser(description='File that executes model training for DEC clustering')


## Dataset
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')

## GPU
parser.add_argument('--cuda_number', type=int, default=4,
                    help='use of cuda (default: 6)')

## Training
parser.add_argument('--starting_epoch', type=int, default=10,
                    help='Starting epoch for KL loss (default: 10)')
parser.add_argument('--base_multiplier', default=4e9, type=float,
                    help='Multiplier for KL loss (default: 4e3)')
parser.add_argument('--lr', default=0.009, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--clusters', default=10, type=int, 
                    help='Amount of cluster centroids (default 10)')
parser.add_argument('--exponent', default=2, type=float,
                    help='Exponent in target distribution for DEC (default: 2)')
parser.add_argument('--include_kldivergence', default=True, type=bool,
                    help='Wether KL loss should be included (default: True)')

## Others
parser.add_argument('--verbose', default=0, type=int,
                    help='print extra information at every epoch.(default: 0)')

args = parser.parse_args()


seed = args.seed
set_random_seed(seed)
torch.cuda.is_available()
cuda_number = args.cuda_number

torch.cuda.set_device(f"cuda:{cuda_number}")



# loading the SENSORIUM+ dataset
pre = "/usr/users/agecker/datasets/sensorium_2022_pictures/real_dataset/"
filenames = [f"{pre}{i}/" for i in os.listdir(pre)]

dataset_fn = "sensorium.datasets.static_loaders"
dataset_config = {
    "paths": filenames,
    "normalize": True,
    "include_behavior": True,
    "include_eye_position": True,
    "batch_size": 128,
    "scale": 0.25,
    "seed": seed,
}

dataloaders = get_data(dataset_fn, dataset_config)

model_fn = "sensorium.models.stacked_core_full_gauss_readout"
model_config = {
    "pad_input": False,
    "stack": -1,
    "layers": 4,
    "input_kern": 9,
    "gamma_input": 6.3831,
    "gamma_readout": 10,
    "feature_reg_weight": 10,
    "hidden_kern": 7,
    "hidden_channels": 128,
    "depth_separable": True,
    "grid_mean_predictor": {
        "type": "cortex",
        "input_dimensions": 2,
        "hidden_layers": 1,
        "hidden_features": 30,
        "final_tanh": True,
    },
    "init_sigma": 0.1,
    "init_mu_range": 0.3,
    "gauss_type": "full",
    "shifter": True,
    "regularizer_type": "adaptive_log_norm",
    "final_batchnorm_scale": False,
}

trainer_fn = "sensorium.training.standard_trainer"

starting_epoch = args.starting_epoch
base_multiplier = args.base_multiplier
clusters = args.clusters
exponent = args.exponent
include_kldivergence = args.include_kldivergence


if include_kldivergence:
    path_ending = f'KL_EM_exp_{exponent}_cluster_{clusters}_mult_{base_multiplier}_reg_adlognorm_se{starting_epoch}'
else:
    path_ending = f'without_KL_sedd_{seed}'

model = get_model(
    model_fn=model_fn,
    model_config=model_config,
    dataloaders=dataloaders,
    seed=seed,
)

trainer_config = {
    "max_iter": 200,
    "verbose": False,
    "lr_decay_steps": 4,
    "avg_loss": False,
    "lr_init": 0.009,
    "base_multiplier": base_multiplier,
    "device": f"cuda:{cuda_number}",
    "wandb_model_congfig": model_config,
    "wandb_dataset_config": dataset_config,
    "wandb_project": "Model_without_rotation",
    "wandb_name": f"{path_ending}",
    "include_kldivergence": include_kldivergence,
    "cluster_number": clusters,
    "use_wandb": True,
    "dec_starting_epoch": starting_epoch,
    'exponent': exponent,
}
if include_kldivergence:
    trainer = get_trainer(trainer_fn=trainer_fn, trainer_config=trainer_config)
    (
        validation_score,
        trainer_output,
        cluster_centers_np,
        predicted,
        state_dict,
    ) = trainer(model, dataloaders, seed=seed)
    print(cluster_centers_np)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        script_directory,
        "cluster_centers",
        f"cluster_centers_{path_ending}.npy",
    )
    np.save(save_path, cluster_centers_np)

    # Base the save path on the script's location
    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        script_directory,
        "model_checkpoints",
        f"sensorium_model_dec_{path_ending}.pth",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    save_path_predicted = os.path.join(
        script_directory,
        "predictions",
        f"predictions_dec_{path_ending}.pt",
    )
    os.makedirs(os.path.dirname(save_path_predicted), exist_ok=True)
    torch.save(predicted, save_path_predicted)
else:
    trainer = get_trainer(trainer_fn=trainer_fn, trainer_config=trainer_config)
    (
        validation_score,
        trainer_output,
        state_dict,
    ) = trainer(model, dataloaders, seed=seed)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        script_directory,
        "model_checkpoints",
        f"sensorium_model_dec_{path_ending}.pth",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

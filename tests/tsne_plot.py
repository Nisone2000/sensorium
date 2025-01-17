import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import torch
from sklearn.manifold import TSNE

import neuralpredictors

warnings.filterwarnings("ignore")

from nnfabrik.builder import get_data, get_model, get_trainer

dec_starting_epcoch = 5
dec_wamup_epcoh = 5


script_directory = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(
    script_directory,
    f"model_checkpoints/sensorium_model_dec_se{dec_starting_epcoch}_we{dec_wamup_epcoh}.pth",
)
if os.path.isfile(directory_path):
    print(f"The directory '{directory_path}' exists.")
else:
    print(f"The directory '{directory_path}' does not exist.")

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
}

dataloaders = get_data(dataset_fn, dataset_config)

model_fn = "sensorium.models.stacked_core_full_gauss_readout"
model_config = {
    "pad_input": False,
    "stack": -1,
    "layers": 4,
    "input_kern": 9,
    "gamma_input": 6.3831,
    "gamma_readout": 0.0076,
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
}
model = get_model(
    model_fn=model_fn,
    model_config=model_config,
    dataloaders=dataloaders,
    seed=42,
)


# Base the save path on the script's location
script_directory = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(
    script_directory,
    f"model_checkpoints/sensorium_model_dec_se{dec_starting_epcoch}_we{dec_wamup_epcoh}.pth",
)
predictions = (
    torch.load(
        f"/user/ninasophie.nellen/sensorium/tests/predictions/predictions_dec_se{dec_starting_epcoch}_we{dec_wamup_epcoh}.pt"
    )
    .cpu()
    .numpy()
)

# Save the model
model.load_state_dict(torch.load(save_path))

# model.load_state_dict(torch.load("./model_checkpoints/sensorium_model_dec_se31_we31.pth"))

color_palette = sns.color_palette("husl", 7)
features_list = []  # Create an empty list to store features
random_indices = [0] * 7
random_indices_predicted = []
features_subset = [None] * 7
colors = []
i = 0
features_shape_previous = 0
for k, readout in model.readout.items():
    features = readout.features.cpu().detach().squeeze().T.numpy()
    random_indices[i] = np.random.choice(features.shape[0], size=2000, replace=False)
    random_indices_predicted.append(
        [arr + features_shape_previous for arr in random_indices[i]]
    )
    features_shape_previous = features.shape[0]
    features_subset[i] = features[random_indices[i], :]
    features_list.append(features)  # Append the features to the list
    print(f"Features for readout {k}: {features.shape}")
    colors.extend(
        [color_palette[i]] * features_subset[i].shape[0]
    )  # Extend colors list with the same color
    i = i + 1

random_indices_predicted = np.concatenate(random_indices_predicted)
predictions_subset = predictions[random_indices_predicted]
print(predictions_subset.shape)

# Combine all subsets into a single array for t-SNE
combined_features = np.vstack(features_subset)
n = combined_features.shape[0]

perplexity_values = [n / 100]
# [30, 50, 100, n/100]
learning_rates = [n / 12]
# [10,100,1000,n/12]
lr = n / 12

# Create a plot for each perplexity value
plt.figure()

for i, perplexity in enumerate(perplexity_values):
    # for j, lr in enumerate(learning_rates):
    # Apply t-SNE
    tsne = TSNE(
        n_components=2, perplexity=perplexity, learning_rate=lr, random_state=42
    )  # Adjust perplexity as needed
    embedded_data = tsne.fit_transform(combined_features)

    # Create a subplot for each perplexity
    plt.subplot(2, 3, i + 1)  # Correct subplot indexing
    plt.scatter(
        embedded_data[:, 0], embedded_data[:, 1], c=predictions_subset, alpha=0.2, s=5
    )
    plt.title(f"Perplexity={perplexity}")

# Adjust layout for better spacing
plt.tight_layout()

handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"Readout {i + 1}",
        markerfacecolor=color_palette[i],
        markersize=5,
    )
    for i in range(7)
]
plt.legend(
    handles=handles,
    title="Readouts",
    loc="upper left",
    bbox_to_anchor=(1, 1),
    frameon=True,
)
plt.savefig(
    f"tsne_model_without_rotation_se{dec_starting_epcoch}_we{dec_wamup_epcoh}.png",
    dpi=300,
    bbox_inches="tight",
)  # Save as PNG with hi
plt.show()

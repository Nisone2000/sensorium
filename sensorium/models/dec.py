# code used from https://github.com/vlukiyanov/pt-dec/blob/master/ptdec/dec.py
# and replaced DAE with the network readouts.

import torch
import torch.nn as nn

from .cluster import ClusterAssignment


class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        model_readouts: torch.nn.Module,
        alpha: float = 1.0,
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the ClusterAssignment stage. Instead of using an
        Autoencoder we use the model readouts for the ClusterAssignment.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.model_readouts = model_readouts
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Get embeddings from the readout model and compute the cluster assignments
        embeddings = self.readout_model(batch)
        cluster_assignments = self.assignment(embeddings)
        return cluster_assignments

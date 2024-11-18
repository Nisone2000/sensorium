from functools import partial

import numpy as np
import torch
from nnfabrik.utility.nn_helpers import set_random_seed
from sklearn.cluster import KMeans
from torch.nn import KLDivLoss
from tqdm import tqdm

import wandb
from neuralpredictors.measures import modules
from neuralpredictors.training import (LongCycler, MultipleObjectiveTracker,
                                       early_stopping)

from ..models.dec import DEC
from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss


def standard_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    use_wandb=True,
    wandb_name=None,
    wandb_model_config=None,
    wandb_dataset_config=None,
    track_training=False,
    detach_core=False,
    deeplake_ds=False,
    save_checkpoints=True,
    checkpoint_save_path="../../tests/model_checkpoints/sensorium_p_rotation_model_dec_",
    chpt_save_step=15,
    include_kldivergence=True,
    cluster_number=10,
    alpha=1.0,
    dec_starting_epoch=10,
    kmeans_init=20,
    **kwargs,
):
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        cb: whether to execute callback function
        track_training: whether to track and print out the training progress

        cluster_number: Give number of clusters for DEC clustering algortihm
        alpha: alpha used for calculation of soft assignment
        dec_starting_epoch: Epoch at which we start the initialisation for the cluster centroids for dec clustering
        kmeans_init: number of iterations for kmeans for cluster initialisation
        **kwargs:

    Returns:

    """

    def get_multiplier(epoch, base_multiplier=1e8):
        if epoch < 14:
            return 0
        else:
            return base_multiplier

    def soft_assignments(encoded_features, cluster_centers, alpha=alpha):
        # Compute soft assingments q_ij as described in DEC paper (1)
        norm_squared = torch.sum(
            (encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, 2
        )
        assignments = 1.0 / (1.0 + (norm_squared / alpha))
        assignments = assignments ** ((alpha + 1) / 2)
        return assignments / torch.sum(assignments, dim=1, keepdim=True)

    def target_distribution(batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch**2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def full_objective(model, dataloader, data_key, *args, **kwargs):
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        regularizers = int(
            not detach_core
        ) * model.core.regularizer() + model.readout.regularizer(data_key)

        tot_main_loss = loss_scale * criterion(
            model(args[0].to(device), data_key=data_key, **kwargs),
            args[1].to(device),
        )
        return (tot_main_loss + regularizers), (tot_main_loss, regularizers)

    ##### Model training ####################################################################################################

    """
    if include_kldivergence:
        model = DEC(cluster_number=cluster_number, hidden_dimension=128, model=model)
    """
    model.to(device)
    set_random_seed(seed)
    model.train()

    kldiv_criterion = KLDivLoss(
        size_average=False
    )  # losses are summed for each minibatch
    criterion = getattr(modules, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    if include_kldivergence:
        cluster_centers = torch.zeros(
            cluster_number, 128, dtype=torch.float, requires_grad=True
        )
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + [cluster_centers],  # Combine model params and additional tensor
            lr=lr_init,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    if track_training:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model,
                dataloaders["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model,
                dataloaders["validation"],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    if use_wandb:
        # initalise wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr_init,
                "architecture": wandb_model_config,
                "dataset": wandb_dataset_config,
                "cur_epochs": max_iter,
                "starting epoch": epoch,
                "lr_decay_steps": lr_decay_steps,
                "lr_decay_factor": lr_decay_factor,
                "min_lr": min_lr,
            },
        )
        # metrics represent any value I want to track, if they're hidden, they're not displayed on default cisualisation
        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)

    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        tracker=tracker,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):

        if include_kldivergence and epoch == dec_starting_epoch:
            # TODO: include hidden dimension
            kmeans = KMeans(
                n_clusters=cluster_number, n_init=kmeans_init, random_state=seed
            )
            model.train()
            feature_list = []
            features_subset = []
            random_indices = {}
            # form initial cluster centres

            for k, readout in model.readout.items():
                features = readout.features.cpu().detach().squeeze().T.numpy()
                feature_list.append(np.array(features))
                random_index = np.random.choice(
                    features.shape[0], size=int(features.shape[0] / 5), replace=False
                )
                random_indices[k] = random_index
                features_subset.append(features[random_indices[k], :])
            features = np.vstack(feature_list)
            features_subset = np.vstack(features_subset)
            predicted = kmeans.fit_predict(features)
            # predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
            # _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
            cluster_centers = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
            )
            cluster_centers = cluster_centers.to(device, non_blocking=True)

            """with torch.no_grad():
                # initialise the cluster centers
                model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
            """

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        epoch_loss = 0
        epoch_loss_main = 0
        epoch_loss_reg = 0
        epoch_loss_kldiv = 0

        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):

            batch_args = list(data)
            batch_kwargs = data._asdict() if not isinstance(data, dict) else data
            loss, loss_parts = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )
            loss.backward()
            epoch_loss += loss.detach()
            epoch_loss_main += loss_parts[0].detach()
            epoch_loss_reg += loss_parts[1].detach()
            cluster_centers_list = []
            if (batch_no + 1) % optim_step_count == 0:
                # TODO maybe remove the hidden dimensions
                if include_kldivergence and epoch >= dec_starting_epoch:
                    kldiv_loss = torch.zeros(1).to(device)
                    features_subset = []
                    feature_list = []
                    for k, readout in model.readout.items():
                        features = readout.features.detach().squeeze()
                        features_subset.append(features[:, random_indices[k]])
                        feature_list.append(features)

                    features_subset = torch.cat(features_subset, dim=1)
                    feature_list = torch.cat(feature_list, dim=1)
                    output = soft_assignments(feature_list, cluster_centers)
                    target = target_distribution(output).detach()
                    """
                    if (target<= 0).any() or (output < 0).any():
                        min_target = target.min()
                        min_output = output.min()

                        # Determine the required shift
                        shift = max(0, -min_target.item(), -min_output.item()) + 1e-5  # A small epsilon to avoid zeros
                        shifted_targets =target + shift
                        shifted_output = output + shift
                        target = shifted_targets / shifted_targets.sum(dim=-1, keepdim=True)
                        output = shifted_output / shifted_output.sum(dim=-1, keepdim=True)
                    """
                    kldiv_loss = (
                        get_multiplier(epoch)
                        * kldiv_criterion(output.log(), target)
                        / output.shape[0]
                    )
                    kldiv_loss.backward()
                    epoch_loss_kldiv += kldiv_loss.detach()
                    epoch_loss += kldiv_loss.detach()
                    cluster_centers_list.append(cluster_centers.cpu().detach())
                optimizer.step()
                optimizer.zero_grad()

            ## after - epoch-analysis
            """
        if save_checkpoints:
            if epoch % chpt_save_step == 0:
                torch.save(
                    model.state_dict(), f"{checkpoint_save_path}epoch_{epoch}.pth"
                ) 
        """
        validation_correlation = get_correlations(
            model,
            dataloaders["validation"],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=deeplake_ds,
        )
        val_loss, val_loss_parts = full_objective(
            model,
            dataloaders["validation"],
            data_key,
            *batch_args,
            **batch_kwargs,
            detach_core=detach_core,
        )
        print(
            f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
        )
        print(
            f"EPOCH={epoch}  validation_correlation={validation_correlation}  Epoch Train loss Kullback-Leibler-divergence={epoch_loss_kldiv}"
        )

        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "Epoch Train loss main": epoch_loss_main,
                "Epoch Train loss regularizers": epoch_loss_reg,
                "Epoch Train loss Kullback-Leibler-divergence": epoch_loss_kldiv,
                "Batch": batch_no,
                "Epoch": epoch,
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch validation loss main": val_loss_parts[0],
                "Epoch validation loss regularizers": val_loss_parts[1],
                # "Epoch validation loss Kullback-Leibler-divergence": val_loss_parts[2],
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if include_kldivergence:
        soft_assignments_list = []
        for k, readout in model.readout.items():
            features = readout.features.detach().squeeze()
            soft_assignments_list.append(soft_assignments(features, cluster_centers))
        predicted = torch.cat(soft_assignments_list).max(1)[1]
        # append final cluster_centers
        cluster_centers_list.append(cluster_centers.cpu().detach())
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
    )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)

    return score, output, cluster_centers_list, model.state_dict()

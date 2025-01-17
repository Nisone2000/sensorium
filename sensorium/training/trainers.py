from functools import partial

import numpy as np
import torch
from neuralpredictors.measures import modules
from neuralpredictors.training import (
    LongCycler,
    MultipleObjectiveTracker,
    early_stopping,
)
from nnfabrik.utility.nn_helpers import set_random_seed
from sklearn.cluster import KMeans
from torch.nn import KLDivLoss
from tqdm import tqdm

import wandb

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
    lr_decay_steps=5,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    use_wandb=True,
    wandb_name=None,
    wandb_project="Rotation_test",
    wandb_entity="ninasophie-nellen-g-ttingen-university",
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
    dec_starting_epoch=5,
    dec_warumup_epoch=10,
    kmeans_init=20,
    base_multiplier=4e3,
    subsamples=2000,
    exponent=2,
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
        dec_warumup_epoch: Epoch at which we have the full regularizer for KL loss
        base_multiplier: multiplier to get KL to same order of magnitude as Poisson loss
        kmeans_init: number of iterations for kmeans for cluster initialisation
        subsamples: number of subsamples used for clustering
        exponent: The exponent for the target distribution for DEC
        **kwargs:

    Returns:

    """

    def get_multiplier(epoch, base_multiplier=4e3):
        """Multiplier to scale KL loss in same order of magnitude as main loss
        To avoid hard peek aat starting epoch we include a warm-up phase s.t. the loss can increase slower
        """
        if epoch < dec_starting_epoch:
            return 0
        else:
            return base_multiplier
        """elif dec_warumup_epoch == dec_starting_epoch:
            return base_multiplier
            elif dec_warumup_epoch >= epoch >= dec_starting_epoch:
            return (
                base_multiplier
                * (epoch - dec_starting_epoch)
                / (dec_warumup_epoch - dec_starting_epoch)
            )"""

    def soft_assignments(encoded_features, cluster_centers, alpha=alpha):
        """Compute soft assingments q_ij as described in DEC paper (1)
        q_ij = (1+ ||z_i - \mu_j||^2/a)^(-(a+1)/2) / (sum_j'((1+ ||z_i - \mu_j'||^2/a)^(-(a+1)/2)))
        """
        norm_squared = torch.sum(
            (encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, 2
        )
        assignments = 1.0 / (1.0 + (norm_squared / alpha))
        assignments = assignments ** ((alpha + 1) / 2)
        return assignments / torch.sum(assignments, dim=1, keepdim=True)

    def target_distribution(batch: torch.Tensor, exponent=exponent) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        p_ij = (q_ij^2/f_j) / sum_j'(q_ij'^2/f_j')  f_j =sum_i(q_ij)

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch**exponent) / torch.sum(batch, 0)
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

    # losses are summed for each batch
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
        k = list(model.readout.keys())[0]
        dim = model.readout[k].features.shape[1]
        dtype = model.readout[k].features.dtype
        cluster_centers = torch.nn.Parameter(
            torch.zeros(
                cluster_number,
                dim,
                dtype=dtype,
                device=device,
            ),
            requires_grad=True,
        )  # Wrap as Parameter
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
        wandb.run.log_code(".")
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
            cluster_centers_list = []
            kmeans = KMeans(
                n_clusters=cluster_number, n_init=kmeans_init, random_state=seed
            )
            feature_list = []
            features_subset = []
            random_indices = {}
            # form initial cluster centres
            with torch.no_grad():
                for k, readout in model.readout.items():
                    features = readout.features.cpu().detach().squeeze().T.numpy()
                    feature_list.append(np.array(features))
                    # TODO remove subset
                    """
                    rng = np.random.default_rng(seed)
                    random_index = rng.choice(
                        features.shape[0], size=int(subsamples), replace=False
                    )
                    random_indices[k] = random_index
                    features_subset.append(features[random_indices[k], :])
                    """
                features = np.vstack(feature_list)
                # features_subset = np.vstack(features_subset)
                predicted = kmeans.fit_predict(features)

            print("Cluster centers old", cluster_centers)
            cluster_centers.data = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=device
            )
            print("Cluster centers new: ", cluster_centers)

            """with torch.no_grad():
                # initialise the cluster centers
                model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
            """

        model.train()
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
        epoch_loss_kldiv_without_scaling = 0

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
            if (batch_no + 1) % optim_step_count == 0:
                # TODO maybe remove the hidden dimensions
                if include_kldivergence and epoch >= dec_starting_epoch:
                    kldiv_loss = torch.zeros(1).to(device)
                    features_subset = []
                    feature_list = []
                    for k, readout in model.readout.items():
                        features = readout.features.squeeze()
                        # features_subset.append(features[:, random_indices[k]])
                        feature_list.append(features)

                    # features_subset = torch.cat(features_subset, dim=1)
                    feature_list = torch.cat(feature_list, dim=1)
                    output = soft_assignments(feature_list, cluster_centers)
                    print('Shape of Q matrix: ', output.shape)
                    print('column_sums for Q', torch.sum(output, dim=0))

                    # detach targets to treet them as pseudolabels for clusters
                    target = target_distribution(output).detach()
                    print('Shape of P matrix: ', target.shape)
                    print('column_sums for P ', torch.sum(target, dim=0))

                    # To avoid underflow issues when computing this quantity, this loss expects the argument input in the log-space.
                    # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                    kldiv_loss = kldiv_criterion(output.log(), target)
                    kldiv_loss.backward()
                    epoch_loss_kldiv += (
                        get_multiplier(epoch, base_multiplier) * kldiv_loss.detach()
                    )
                    epoch_loss_kldiv_without_scaling += kldiv_loss.detach()
                    epoch_loss += (
                        get_multiplier(epoch, base_multiplier) * kldiv_loss.detach()
                    )
                    """
                    if cluster_centers.grad is not None:
                        print("Gradients for cluster_centers exist.")
                        print("Gradient values:", cluster_centers.grad)
                    else:
                        print("No gradients for cluster_centers. Check if it is part of the computational graph.")
                    """
                    with torch.no_grad():
                        cluster_centers_clone = cluster_centers.clone()
                        cluster_centers_list.append(
                            cluster_centers_clone.cpu().detach()
                        )

                # if include_kldivergence and epoch == dec_starting_epoch:
                #    print('cluster centers old', cluster_centers)
                optimizer.step()
                # if include_kldivergence and epoch == dec_starting_epoch:
                #    print('cluster centers updated', cluster_centers)
                optimizer.zero_grad()

        ll = model.core.features.layer3.norm
        # print('model core', model.core)
        # print(ll)
        assert ll.affine == False
        # assert (ll.weight == 1).all() == True
        # assert (ll.bias == 0).all() == True

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
            f"EPOCH={epoch}  validation_correlation={validation_correlation}  Epoch Train loss Kullback-Leibler-divergence={epoch_loss_kldiv_without_scaling}"
        )

        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "Epoch Train loss main": epoch_loss_main,
                "Epoch Train loss regularizers": epoch_loss_reg,
                "Epoch Train loss Kullback-Leibler-divergence": epoch_loss_kldiv,
                "Epoch Train loss KL without scaling": epoch_loss_kldiv_without_scaling,
                "Batch": batch_no,
                "Epoch": epoch,
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch validation loss main": val_loss_parts[0],
                "Epoch validation loss regularizers": val_loss_parts[1],
                "Learning rate": optimizer.param_groups[0]["lr"],
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
        cluster_centers_list.append(cluster_centers.cpu().detach().numpy())
        cluster_centers_np = np.array(cluster_centers_list)
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
    )

    if use_wandb:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)

    if include_kldivergence:
        return score, output, cluster_centers_np, predicted, model.state_dict()
    else:
        return score, output, model.state_dict()

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
    wandb_model_config=None,
    wandb_dataset_config=None,
    wandb_entity='ninasophie-nellen-g-ttingen-university',
    wandb_project='Model_without_rotation',
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
    use_diag_cov=True,
    learn_alpha=True,
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
        use_diag_conv: Bool that indicates wether to use a diagonal covariance matrix or just one value for each cluster in EM step
        learn_alpha: learn alpha or set it as a parameter
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
        """
        Compute soft assingments q_ij as described in DEC paper (1)
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
    
    def soft_assignments_mult(encoded_features, cluster_centers, sigma, alpha=1, p=1):
        sigma_inv = 1.0 / sigma 
        #+1e-8 # (K,)
        diff = (encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) #(N,K,D)
        #print(diff.shape)
        norm_sigma = torch.sum((diff **2 * sigma_inv),2)  
        print('norm_sigma', norm_sigma)
        #det = torch.sqrt(torch.prod(sigma, dim=1))
        log_det = torch.sum(torch.log(sigma), dim=1) 
        det = torch.exp(0.5 * log_det) 
        print('det log ', det)
        det = torch.sqrt(torch.prod(sigma, dim=1)) + 1e-6
        print('det + const', det)
        assignments = 1.0 / (1.0 + (norm_sigma / alpha))
        assignments = (assignments ** ((alpha + p) / 2))/det
        #print('Assignments shape', assignments.shape)
        return assignments / torch.sum(assignments, dim=1, keepdim=True)

    def EM_t_mult(features, resp, cluster_centers, sigma, alpha, d=1):
        sigma_inv = 1.0 / sigma  # (K,)
        diff = (features.T.unsqueeze(1) - cluster_centers.unsqueeze(0))
        norm_sigma = torch.sum((diff **2 * sigma_inv),2)  
        u = ((alpha + d)/(alpha +norm_sigma)).detach()  #ccalculate U shape(N,K)
        print('u', u)
        #print(resp.shape)
        
        ''' M step '''
        numerator = torch.matmul(features,resp*u).T.detach()
        #print(numerator.shape)
        denominator = torch.sum(resp*u, dim=0, keepdim=True).T.detach()
        print(denominator)
        cluster_centers = numerator/denominator

        print('cluster centers', cluster_centers)

        weighted_sq_diff = resp.unsqueeze(2)* u.unsqueeze(2) * (diff ** 2)  # (N, K, D)
        numerator = weighted_sq_diff.sum(dim=0) #(K,D)
        denominator = torch.sum(resp, dim=0, keepdim=True)  # (K,)
        print('denominator', denominator)
        #print('sigma de', denominator.shape)
        sigma = (numerator / denominator.T).detach() 
        
        sigma = torch.clamp(sigma, min=1e-4, max=1e4)

        return cluster_centers, sigma
    
    def EM_t_1D(features, resp, cluster_centers, taus, alpha, d=1):
        norm_squared = torch.sum(
        (features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, dim=2
        )  
        u = (alpha + d)/(alpha +norm_squared*(taus**(-1)))  #ccalculate U shape(N,K)
        print('u', u.shape)
        print(resp.shape)
        ''' M step '''
        numerator = torch.matmul(features,resp*u).T.detach()
        print(numerator.shape)
        denominator = torch.sum(resp*u, dim=0, keepdim=True).T.detach()
        print(denominator.shape)
        cluster_centers = numerator/denominator

        weighted_sums = torch.sum(resp* u * norm_squared, dim=0) 
        #print('WS', weighted_sums.shape)
        taus = (weighted_sums / torch.sum(resp, dim=0, keepdim=True)).detach()
        #print('Tau', taus.shape)
        return cluster_centers, taus


    def soft_assignments_1D(encoded_features, cluster_centers, tau, alpha=1, p=1):
        norm_squared = torch.sum(
            (encoded_features.T.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, 2
        )
        assignments = 1.0 / (1.0 + (norm_squared / (alpha * tau)))
        assignments = (assignments ** ((alpha + p) / 2))/(tau**1/2)
        #print('Assignments shape', assignments.shape)
        return assignments / torch.sum(assignments, dim=1, keepdim=True)

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
    
    def repulsion_loss(cluster_centers, epsilon=1e-6):
        'Regularization term that penalizes small distances between clusters'
        pairwise_distances = torch.cdist(cluster_centers, cluster_centers, p=2)  # Compute pairwise distances
        pairwise_distances = torch.triu(pairwise_distances, diagonal=1)  # Keep only upper triangle (ignores diagonal)
        return torch.sum(1.0 / (pairwise_distances + epsilon)) 
    
    
    def dec_loss(epoch, base_multiplier, output, target, cluster_centers):
        kldiv_loss = get_multiplier(epoch, base_multiplier) * (kldiv_criterion(output.log(), target))
        regularizer = get_multiplier(epoch, base_multiplier) *  repulsion_loss(cluster_centers)
        return (kldiv_loss + regularizer), (kldiv_loss, regularizer)
                   

    ##### Model training ####################################################################################################


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

    if learn_alpha:
        alpha = torch.nn.Parameter(torch.tensor(1.0, device=device, requires_grad=True))
        optimizer = torch.optim.Adam(list(model.parameters()) + [alpha], lr=lr_init)
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
    batch_no_total = 0
    kldiv_list = []
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
            # form initial cluster centres
            with torch.no_grad():
                for k, readout in model.readout.items():
                    features = readout.features.cpu().detach().squeeze().T.numpy()
                    feature_list.append(np.array(features))

                features = np.vstack(feature_list)
                predicted = kmeans.fit_predict(features)
                '''
                wcs = []
                # caclculate within cluster variance
                for k in range(cluster_number):
                    cluster_points = features[predicted == k] 
                    if cluster_points.shape[0] > 0: 
                        wcs.append(np.mean(np.linalg.norm(cluster_points - kmeans.cluster_centers_[k], axis=1) ** 2))
                    else:
                        wcs.append(0)
                wcs = np.array(wcs)
                print("Within-Cluster Variance:", wcs)
                np.save(f'/user/ninasophie.nellen/sensorium/tests/wcv/wcv_exponent_{exponent}_{base_multiplier}_se_{dec_starting_epoch}.npy', wcs)
                '''
            cluster_centers = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float, device=device
            )
            if use_diag_cov:
                p = features.shape[1]
                sigma = torch.zeros((cluster_number,p), device=device)
                for k in range(cluster_number):
                    cluster_points = torch.from_numpy(features[predicted == k]).to(device) 
                    print(f'Points for cluster {k}: {cluster_points.shape[0]}')
                    if len(cluster_points) > 0:
                        sigma[k] = torch.var(cluster_points, dim=0, unbiased=True) +1e-6
                print(sigma)

            else: 
                sigma = torch.zeros(cluster_number, device=device)
                for k in range(cluster_number):
                    cluster_points = torch.from_numpy(features[predicted == k]).to(device)  
                    if len(cluster_points) > 0:
                        sigma[k] = torch.mean(torch.sum((cluster_points - cluster_centers[k])**2,1))
                sigma = sigma.unsqueeze(0)
                p=1
            print('p',p)
            if learn_alpha:
                alpha.data = torch.tensor(1.0, dtype=torch.float, device=device)

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
        epoch_kldiv_loss_regularizer = 0
        
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_total +=1
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
                    feature_list = []
                    for k, readout in model.readout.items():
                        features = readout.features.squeeze()
                        feature_list.append(features)

                    # features_subset = torch.cat(features_subset, dim=1)
                    feature_list = torch.cat(feature_list, dim=1)
                    if use_diag_cov:
                        output = soft_assignments_mult(features, cluster_centers, sigma, alpha,p)
                    else:
                        output = soft_assignments_1D(features, cluster_centers, sigma, alpha,p)
                    #print('Shape of Q matrix: ', output.shape)
                    #print('Row sums for Q', torch.sum(output, dim=1))

                    # detach targets to treat them as pseudolabels for clusters
                    target = target_distribution(output, exponent)

                    # To avoid underflow issues when computing this quantity, this loss expects the argument input in the log-space.
                    # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                    kldiv_loss, kldiv_loss_parts = dec_loss(epoch, base_multiplier, output, target, cluster_centers)
                    kldiv_loss.backward()
                    epoch_loss_kldiv += kldiv_loss.detach()
                    epoch_loss_kldiv_without_scaling += kldiv_loss_parts[0].detach() / get_multiplier(epoch, base_multiplier)
                    epoch_kldiv_loss_regularizer += kldiv_loss_parts[1].detach()
                    epoch_loss += kldiv_loss.detach()

                    with torch.no_grad():
                        cluster_centers_list.append(cluster_centers.cpu().detach())
                        kldiv_list.append(kldiv_loss.cpu() / get_multiplier(epoch, base_multiplier))

                    if use_diag_cov:
                        cluster_centers, sigma = EM_t_mult(features, output, cluster_centers, sigma, alpha, p)
                    else:
                        cluster_centers, sigma = EM_t_1D(features, output, cluster_centers, sigma, alpha, p)
                    # Normalize the cluster centers such that they represent the real mean of the clusters
                    '''
                    numerator = torch.matmul(feature_list,output).T.detach()
                    denominator = torch.sum(output, dim=0, keepdim=True).T.detach()
                    cluster_centers = numerator/denominator
                    '''
                    print(sigma)

                optimizer.step()
                optimizer.zero_grad()
            
            '''
            if use_wandb:
                wandb_dict = {
                    "Epoch Train loss": epoch_loss,
                    "Epoch Train loss main": epoch_loss_main,
                    "Epoch Train loss regularizers": epoch_loss_reg,
                    "Epoch Train loss Kullback-Leibler-divergence": epoch_loss_kldiv,
                    "Epoch Train loss KL without scaling": epoch_loss_kldiv_without_scaling,
                    "Batch": batch_no_total,
                    "Epoch": epoch,
                    "Learning rate": optimizer.param_groups[0]["lr"],
                    # "Epoch validation loss Kullback-Leibler-divergence": val_loss_parts[2],
                }
                wandb.log(wandb_dict)
            '''
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
                "Epoch Train loss KL without scaling main": epoch_loss_kldiv_without_scaling,
                'Epoch Train loss KL regularizers': epoch_kldiv_loss_regularizer,
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
        kldiv_list_np = np.array(kldiv_list)
    tracker.finalize() if track_training else None
    np.save(f'/user/ninasophie.nellen/sensorium/tests/cluster_centers/kldiv_repulsion_loss_exponent_{exponent}_{base_multiplier}_se_{dec_starting_epoch}.npy', kldiv_list_np)

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

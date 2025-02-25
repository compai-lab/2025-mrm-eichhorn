import copy
import glob
import os.path
import inspect
import numpy as np
import torch
from core.Trainer import Trainer
from time import time
import wandb
import logging
from torchinfo import summary
from projects.moco_t2star.utils import *
from data.t2star_loader import generate_masks
from mr_utils.parameter_fitting import T2StarFit


class PTrainer(Trainer):
    def __init__(self, training_params, model, hyper_setup, data, device,
                 log_wandb=True):
        super(PTrainer, self).__init__(
            training_params, model, hyper_setup, data, device, log_wandb)

        for s in self.train_ds:
            input_size = s[8].unsqueeze(1).numpy().shape
            break

        if self.hyper_setup:
            logging.info(
                "[Trainer::init]: HyperNetwork setup is not implemented yet."
            )
        else:
            print(f'Input size of summary is: {input_size}')
            summary(self.model, input_size)

        logging.info("[Trainer::init]: Loading reconstruction model with "
                     "the following parameters:\n{}.".format(
            self.training_params["recon_model"])
        )
        self.recon_model = load_recon_model(
            self.training_params["recon_model"], self.device
        )

        self.early_stop_regularisation = self.training_params.get(
            "early_stop_regularisation", False
        )

        self.keep_center = self.training_params.get("keep_center", False)
        self.exclude_last_echoes = self.training_params.get(
            "exclude_last_echoes", False
        )

        self.finetune = self.training_params.get("finetune", False)
        if self.finetune is not False:
            logging.info("[Trainer::train]: Loading the initial model for "
                         "finetuning.")
            self.initial_model = self._load_initial_model()


    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Trains the local client with the option to initialize the model and
        optimizer states.

        Parameters
        ----------
        model_state : dict, optional
            Weights of the global model. If provided, the local model is
            initialized with these weights.
        opt_state : dict, optional
            State of the optimizer. If provided, the local optimizer is
            initialized with this state.
        start_epoch : int, optional
            Starting epoch for training.

        Returns
        -------
        dict
            The state dictionary of the trained local model.
        """

        if model_state is not None:
            self.model.load_state_dict(model_state)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        self.early_stop = False

        self.model.train()
        # predicted_masks = {}

        for epoch in range(self.training_params['nr_epochs']):
            print('Epoch: ', epoch)
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info(
                    "[Trainer::test]: ################ Finished  training "
                    "(early stopping) ################"
                )
                break

            start_time = time()
            batch_loss = {"combined": 0}
            evaluation_metrics = {}
            count_images = 0

            for data in self.train_ds:
                (img_cc_fs, sens_maps, img_cc_fs_gt, img_hrqrmoco, brain_mask,
                 brain_mask_noCSF, filename, slice_num,
                 simulated_data, mask_simulation) = process_input_data(
                    self.device, data
                )
                count_images += img_cc_fs.shape[0]

                # Forward Pass
                self.optimizer.zero_grad()

                if self.finetune is not False:
                    mask_finetune = self.model(slice_num)
                    with torch.no_grad():
                        mask_initial = self.initial_model(slice_num)
                    mask_predicted = torch.min(mask_initial, mask_finetune)
                else:
                    mask_predicted = self.model(slice_num)

                prediction, img_zf, mask_full = perform_reconstruction(
                    img_cc_fs, sens_maps, mask_predicted,
                    self.recon_model, keep_central_point=self.keep_center
                )

                if self.exclude_last_echoes is not False:
                    prediction = prediction[:, :-self.exclude_last_echoes]
                    img_cc_fs = img_cc_fs[:, :-self.exclude_last_echoes]
                    img_cc_fs_gt = img_cc_fs_gt[:, :-self.exclude_last_echoes]

                losses = [0]
                # Physics Loss (Model Fit Error)
                loss_physics = self.criterion_physics(
                    prediction,
                    mask=brain_mask_noCSF[:, None].repeat(1, 12, 1, 1)
                )
                if self.lambda_physics > 0:
                    losses[0] += self.lambda_physics * loss_physics
                    losses.append(loss_physics)

                # Regularisation:
                if self.criterion_reg is not None:
                    if (self.start_epoch_reg is None or
                            (self.start_epoch_reg is not None
                             and epoch >= self.start_epoch_reg)):
                        num_args = len(
                            inspect.signature(self.criterion_reg).parameters)
                        if num_args == 1:
                            loss_reg = self.criterion_reg(mask_predicted)
                            if loss_reg is not None:
                                if self.lambda_reg > 0:
                                    losses[0] += self.lambda_reg * loss_reg
                                losses.append(loss_reg)
                        if num_args == 2:
                            loss_reg = self.criterion_reg(mask_predicted,
                                                          slice_num)
                            if loss_reg is not None:
                                if self.lambda_reg > 0:
                                    losses[0] += self.lambda_reg * loss_reg
                                losses.append(loss_reg)

                if self.criterion_2nd_reg is not None:
                    if (self.start_epoch_2nd_reg is None or
                            (self.start_epoch_2nd_reg is not None
                             and epoch >= self.start_epoch_2nd_reg)):
                        num_args = len(
                            inspect.signature(self.criterion_2nd_reg).parameters)
                        if num_args == 1:
                            loss_2nd_reg = self.criterion_2nd_reg(mask_predicted)
                        elif num_args == 2:
                            loss_2nd_reg = self.criterion_2nd_reg(mask_predicted,
                                                                  slice_num)
                        else:
                            loss_2nd_reg = None
                            logging.info(
                                "[Trainer::train]: The 2nd regularisation "
                                "function should have one or two arguments."
                            )
                        if loss_2nd_reg is not None:
                            if self.lambda_2nd_reg > 0:
                                losses[0] += self.lambda_2nd_reg * loss_2nd_reg
                            losses.append(loss_2nd_reg)

                # Backward Pass on the combined loss
                losses[0].backward()
                self.optimizer.step()
                batch_loss = self._update_batch_loss(
                    batch_loss, losses, img_cc_fs.size(0)
                )

                with torch.no_grad():
                    # Evaluation metrics (here without registration):
                    if epoch == 0:
                        img_metrics_motion = calculate_img_metrics(
                            target=img_cc_fs_gt,
                            data=img_cc_fs,
                            bm=brain_mask[:, None].repeat(1, img_cc_fs_gt.shape[1], 1, 1),
                            metrics_to_be_calc=["SSIM_magn", "PSNR_magn"],
                            include_brainmask=True)
                        t2star_metrics_motion = calculate_t2star_metrics(
                            target=img_cc_fs_gt,
                            data=img_cc_fs,
                            bm=brain_mask_noCSF[:, None].repeat(1, img_cc_fs_gt.shape[1], 1, 1),
                            metrics_to_be_calc=["T2s_MAE"])
                        print("Metrics for motion-corrupted image: ")
                        print(img_metrics_motion, t2star_metrics_motion)

                    img_metrics_pred = calculate_img_metrics(
                        target=img_cc_fs_gt,
                        data=prediction,
                        bm=brain_mask[:, None].repeat(1, img_cc_fs_gt.shape[1], 1, 1),
                        metrics_to_be_calc=["SSIM_magn", "PSNR_magn"],
                        include_brainmask=True)
                    self._update_batch_metrics(
                        evaluation_metrics, img_metrics_pred, img_cc_fs.size(0)
                    )

                    t2star_metrics_pred = calculate_t2star_metrics(
                        target=img_cc_fs_gt,
                        data=prediction,
                        bm=brain_mask_noCSF[:, None].repeat(1, img_cc_fs_gt.shape[1], 1, 1),
                        metrics_to_be_calc=["T2s_MAE"])
                    self._update_batch_metrics(
                        evaluation_metrics, t2star_metrics_pred,
                        img_cc_fs.size(0)
                    )

                    if simulated_data:
                        mask_metrics_pred = calculate_mask_metrics(
                            target=mask_simulation,
                            data=mask_predicted,
                            metrics_to_be_calc=["MAE"]
                        )
                        self._update_batch_metrics(
                            evaluation_metrics, mask_metrics_pred,
                            img_cc_fs.size(0)
                        )

                    for slice in [10, 11, 15, 16, 20, 21, 25, 30]:
                        if slice in slice_num:
                            ind = np.where(detach_torch(slice_num) == slice)[0][0]

                            last_epoch = self.training_params['nr_epochs']-1

                            second_mask_example = (mask_simulation[ind]
                                                   if simulated_data else None)
                            third_mask_example = (mask_finetune[ind]
                                                  if self.finetune is not False else None)

                            if epoch % 20 == 0 or epoch == last_epoch:
                                log_images_to_wandb(
                                    prediction_example=prediction[ind],
                                    ground_truth_example=img_cc_fs_gt[ind],
                                    motion_example=img_cc_fs[ind],
                                    mask_example=mask_full[ind],
                                    second_mask_example=second_mask_example,
                                    third_mask_example=third_mask_example,
                                    wandb_log_name="Train_Example_Images/",
                                    slice=slice,
                                    data_types=["magn"]
                                )

                            if epoch % 100 == 0 or epoch == last_epoch:
                                calc_t2star = T2StarFit(dim=3)
                                t2star_fs = detach_torch(calc_t2star(
                                    img_cc_fs[ind], mask=brain_mask[ind]
                                ))
                                t2star_fs_gt = detach_torch(calc_t2star(
                                    img_cc_fs_gt[ind], mask=brain_mask[ind]
                                ))
                                t2star_pred = detach_torch(calc_t2star(
                                    prediction[ind], mask=brain_mask[ind]
                                ))
                                log_t2star_maps_to_wandb(
                                    t2star_pred,
                                    t2star_fs_gt,
                                    t2star_fs,
                                    wandb_log_name="Train_Example_T2star/",
                                    slice=slice,
                                )

            self._track_epoch_loss(epoch, batch_loss, start_time, count_images)
            self._track_epoch_metrics(epoch, evaluation_metrics, count_images)

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(),
                        'optimizer_weights': self.optimizer.state_dict(),
                        'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val',
                      self.optimizer.state_dict(), epoch, batch_loss)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None,
             epoch=0, train_batch_loss=None):
        """
        Tests the local client.

        Parameters
        ----------
        model_weights : dict
            Weights of the global model.
        test_data : DataLoader
            Test data for evaluation.
        task : str, optional
            Task identifier (default is 'Val').
        opt_weights : dict, optional
            Optimal weights (default is None).
        epoch : int, optional
            Current epoch number (default is 0).
        """

        if task == 'Val':
            epoch_val_loss = self.get_epoch_val_loss(train_batch_loss, epoch)

            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save({'model_weights': model_weights,
                            'optimizer_weights': opt_weights,
                            'epoch': epoch},
                           self.client_path + '/best_model.pt')
                self.best_weights = model_weights
                self.best_opt_weights = opt_weights
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                wandb.log({
                    "Train_LearningRate": self.lr_scheduler.get_last_lr()[0],
                    '_step_': epoch
                })

    def _load_initial_model(self):
        """Load the initial predictions for finetuning."""

        initial_model = copy.deepcopy(self.model)
        initial_weights = glob.glob("{}/{}/**/best_model.pt".format(
            self.finetune["group"], self.finetune["subject"])
        )[0]
        checkpoint = torch.load(initial_weights,
                                map_location=torch.device(self.device))
        initial_model.load_state_dict(checkpoint['model_weights'])
        initial_model.to(self.device)
        initial_model.eval()

        return initial_model

    @staticmethod
    def _update_batch_loss(batch_loss, losses, batch_size):
        """Update batch losses with current losses"""

        batch_loss["combined"] += losses[0].item() * batch_size
        for i in range(1, len(losses)):
            if f"component_{i}" not in batch_loss.keys():
                batch_loss[f"component_{i}"] = 0
            batch_loss[f"component_{i}"] += losses[i].item() * batch_size

        return batch_loss

    @staticmethod
    def _update_batch_metrics(evaluation_metrics, metrics, batch_size):
        """Update evaluation metrics with current metrics"""

        for metric_key in metrics.keys():
            if metric_key not in evaluation_metrics.keys():
                evaluation_metrics[metric_key] = 0
            evaluation_metrics[metric_key] += metrics[metric_key] * batch_size

        return evaluation_metrics

    @staticmethod
    def _track_epoch_loss(epoch, batch_loss, start_time, count_images):
        """Track and log epoch loss."""

        loss_components = batch_loss.keys()
        epoch_loss = {}
        for component in loss_components:
            epoch_loss[component] = (batch_loss[component] / count_images
                                     if count_images > 0
                                     else batch_loss[component])

        end_time = time()
        loss_msg = (
            'Epoch: {} \tTraining Loss: {:.6f} , computed in {} '
            'seconds for {} samples').format(
            epoch, epoch_loss["combined"], end_time - start_time,
            count_images
        )
        print(loss_msg)

        for component in loss_components:
            name = f'Train_Loss_{component.replace("component_", "Comp")}'
            wandb.log({name: epoch_loss[component], '_step_': epoch})

    @staticmethod
    def _track_epoch_metrics(epoch, evaluation_metrics, count_images):
        """Track and log epoch metrics."""

        metrics = evaluation_metrics.keys()
        epoch_metrics = {}
        for metric in metrics:
            epoch_metrics[metric] = (evaluation_metrics[metric] / count_images
                                     if count_images > 0
                                     else evaluation_metrics[metric])

        for metric in metrics:
            name = f'Train_Evaluation_metrics/{metric}'
            wandb.log({name: epoch_metrics[metric], '_step_': epoch})

    def get_epoch_val_loss(self, train_batch_loss, epoch):
        """Get the validation loss for the current epoch."""

        if train_batch_loss is not None:
            if self.early_stop_regularisation:
                if "component_2" in train_batch_loss.keys():
                    if "component_3" in train_batch_loss.keys():
                        epoch_val_loss = (
                                (self.lambda_reg * train_batch_loss["component_2"]
                                 + self.lambda_2nd_reg * train_batch_loss["component_3"])
                                / len(self.train_ds)
                        )
                        print("Using 2nd and 3rd component of the {} losses "
                              "for early stopping. Please confirm that this "
                              "corresponds to the regularisation loss.".format(
                            len(train_batch_loss.keys()) - 1)
                              )
                    else:
                        epoch_val_loss = (train_batch_loss["component_2"]
                                          / len(self.train_ds))
                        if epoch == 0:
                            print("Using 2nd component of the {} losses for "
                                  "early stopping. Please confirm that this "
                                  "corresponds to the regularisation "
                                  "loss.".format(len(train_batch_loss.keys())-1)
                                  )
                else:
                    print("No regularisation loss available.")
                    epoch_val_loss = 0
            else:
                epoch_val_loss = (train_batch_loss["combined"]
                                  / len(self.train_ds))
        else:
            epoch_val_loss = 0
            print("No training loss available for early stopping.")

        return epoch_val_loss

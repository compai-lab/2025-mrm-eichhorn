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
from projects.line_detection_t2star.utils import *


class PTrainer(Trainer):
    def __init__(self, training_params, model, hyper_setup, data, device,
                 log_wandb=True):
        super(PTrainer, self).__init__(
            training_params, model, hyper_setup, data, device, log_wandb)

        if self.hyper_setup:
            logging.info(
                "[Trainer::init]: HyperNetwork setup is not implemented."
            )
        else:
            print(f"Input size of summary is: {training_params['input_size']}")
            summary(self.model, training_params['input_size'], dtypes=[torch.float64])


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
        epoch_losses = []

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
            batch_loss = 0
            evaluation_metrics = {}
            count_images = 0

            for data in self.train_ds:
                (img_cc_fs, sens_maps, _, _, _, _, filename, slice_num,
                 simulated_data, mask_simulation) = process_input_data(
                    self.device, data
                )
                count_images += img_cc_fs.shape[0]
                kspace = process_kspace_linedet(
                    img_cc_fs, sens_maps,
                    coils_channel_dim=self.training_params['coils_channel_dim'],
                    coil_combined=self.training_params['coil_combined']
                )

                # Forward Pass
                self.optimizer.zero_grad()
                prediction = self.model(kspace)

                loss = self.criterion_rec(mask_simulation, prediction)
                loss.backward()
                self.optimizer.step()
                batch_loss += loss.item() * kspace.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            print(
                'Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                    epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(),
                        'optimizer_weights': self.optimizer.state_dict(),
                        'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val',
                      self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None,
             epoch=0):
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

        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {task + '_loss_': 0}
        test_total = 0

        val_image_available = False
        with torch.no_grad():
            for data in test_data:
                (img_cc_fs, sens_maps, _, _, _, _, filename, slice_num,
                 simulated_data, mask_simulation) = process_input_data(
                    self.device, data
                )
                kspace = process_kspace_linedet(
                    img_cc_fs, sens_maps,
                    coils_channel_dim=self.training_params['coils_channel_dim'],
                    coil_combined=self.training_params['coil_combined']
                )

                test_total += img_cc_fs.shape[0]

                # Forward Pass
                prediction = self.test_model(kspace)
                prediction_track = prediction.clone().detach()
                target_mask_track = mask_simulation.clone().detach()

                loss_bce = self.criterion_rec(mask_simulation, prediction)

                metrics[task + '_loss_'] += loss_bce.item() * kspace.size(0)

                if task == 'Val':
                    last_epoch = self.training_params['nr_epochs'] - 1
                    if epoch % 20 == 0 or epoch == last_epoch:
                        for file, slice in zip(detach_torch(filename), detach_torch(slice_num)):
                            if slice in [10, 15]:
                                ind = np.where(detach_torch(slice_num) == slice)[0][0]

                                prediction_example = prediction_track[ind].detach().cpu().numpy().reshape(
                                    -1, 92)
                                target_example = target_mask_track[ind].detach().cpu().numpy().reshape(-1, 92)

                                # reshape to full mask
                                prediction_example = np.rollaxis(
                                    np.tile(prediction_example, (112, 1, 1)), 0, 3)
                                target_example = np.rollaxis(
                                    np.tile(target_example, (112, 1, 1)), 0, 3)

                                # substitute two pixels to get consistent colormap:
                                prediction_example[0, 0, 0] = 0
                                prediction_example[0, 0, 1] = 1
                                target_example[0, 0, 0] = 0
                                target_example[0, 0, 1] = 1

                                prediction_example = prediction_example[0]
                                target_example = target_example[0]

                                if len(np.unique(target_example)) < 3:
                                    thr_prediction = np.zeros_like(prediction_example)
                                    thr_prediction[prediction_example > 0.5] = 1

                                pred = wandb.Image(prediction_example[:, ::-1],
                                                caption='Predicted corruption mask')
                                targ = wandb.Image(target_example[:, ::-1],
                                                caption='Target corruption mask')
                                pred_th = wandb.Image(thr_prediction[:, ::-1],
                                                    caption='Predicted corruption mask (thresholded)')
                                subject = os.path.basename(file).split("_nr_")[0]
                                wandb.log({f'{task}/Examples/{subject}_slice{int(slice[0])}': [pred, pred_th, targ]})


            for metric_key in metrics.keys():
                metric_name = task + '/' + str(metric_key)
                metric_score = metrics[metric_key] / test_total
                wandb.log({metric_name: metric_score, '_step_': epoch})
            wandb.log(
                {'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
            epoch_val_loss = metrics[task + '_loss_'] / test_total
            if task == 'Val':
                print(
                    'Epoch: {} \tValidation Loss: {:.6f} , computed for {} samples'.format(
                        epoch, epoch_val_loss, test_total))
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
                    self.lr_scheduler.step(epoch_val_loss)

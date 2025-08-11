import glob
import yaml
import torch
import numpy as np
import gc
from sklearn.metrics import precision_recall_curve
from data.t2star_loader import *
from dl_utils.config_utils import import_module
from projects.moco_t2star.utils import *
from projects.line_detection_t2star.utils import process_kspace_linedet


def gpu_memory_cleaning(log=False):
    """
    Clean the GPU memory.
    """

    if log:
        print("###################################")
        print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")

    gc.collect()
    torch.cuda.empty_cache()

    if log:
        print("Memory cleaned")
        print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")
        print ("###################################")



def load_masks_into_data_dict(data_dictionary, configuration, subjects,
                              device="cuda"):
    """
    Load the predicted masks for the different experiments into the data_dict.
    """

    data_dictionary['configs_train'] = {}
    data_dictionary['mask_phimo'] = {key: {} for key in configuration['experiments'].keys()}

    for exp_name in configuration['experiments']:
        exp_id = configuration['experiments'][exp_name]["id"]
        if exp_name not in data_dictionary['configs_train']:
            config_path = glob.glob(f"{configuration['checkpoint_path']}weights/"
                                    f"{exp_id}/**/**/config.yaml")[0]

            with open(config_path, 'r') as stream_file:
                data_dictionary['configs_train'][exp_name] = yaml.load(
                    stream_file, Loader=yaml.FullLoader
                )

        for subject in subjects:
            with torch.no_grad():
                if subject not in data_dictionary['mask_phimo'][exp_name]:
                    data_dictionary['mask_phimo'][exp_name][subject] = []

                # load the pretrained model and predict masks:
                pretrained_weights = glob.glob(
                    f"{configuration['checkpoint_path']}weights/"
                    f"{exp_id}/{subject}/**/latest_model.pt")[0]
                model = load_model(pretrained_weights,
                                   configuration['experiments'][exp_name],
                                   device=device)
                indices = torch.tensor(data_dictionary['slices_ind'][subject][:, None]).to(device)
                data_dictionary['mask_phimo'][exp_name][subject] = model(indices).cpu().detach().numpy()

    return data_dictionary


def load_model(pretrained_weights, model_config, device="cuda"):
    """
    Load the pretrained model.
    """

    checkpoint = torch.load(pretrained_weights,
                            map_location=torch.device(device))
    model_class = import_module(model_config['module_name'],
                                model_config['class_name'])
    model = model_class(**(model_config['params']))
    model.load_state_dict(checkpoint['model_weights'])

    return model.to(device).eval()


def perform_phimo_reconstructions(data_dictionary, configuration, subjects,
                                  device="cuda"):
    """
    Perform PHIMO reconstructions with the predicted masks.
    """

    data_dictionary['img_phimo'] = {
        key: {} for key in configuration['experiments'].keys()
    }

    for exp_name in configuration['experiments']:
        recon_model = load_recon_model(
            data_dictionary['configs_train'][exp_name]['recon_model_downstream'],
            device=device
        )
        keep_center = configuration['experiments'][exp_name]['keep_center']
        for subject in subjects:
            # if subject not in data_dictionary['img_phimo'][exp_name]:
            #     data_dictionary['img_phimo'][exp_name][subject] = []

            with torch.no_grad():
                prediction, _, _ = perform_reconstruction(
                    torch.tensor(data_dictionary['img_motion'][subject],
                                 dtype=torch.complex64, device=device),
                    torch.tensor(data_dictionary['sens_maps'][subject],
                                 dtype=torch.complex64, device=device),
                    torch.tensor(data_dictionary['mask_phimo'][exp_name][subject],
                                    dtype=torch.float32, device=device),
                    recon_model,
                    keep_central_point=keep_center
                )
                data_dictionary['img_phimo'][exp_name][subject] = detach_torch(
                    prediction
                )

    return data_dictionary


def perform_orba_reconstructions(data_dictionary, configuration, subjects,
                                 device="cuda"):
    """
    Perform ORBA reconstructions with the predicted masks.
    """

    data_dictionary['img_orba'] = {}
    data_dictionary['mask_orba'] = {}

    recon_model = load_recon_model(
        configuration['orba-settings']['recon_model'], device=device
    )
    random_mask = configuration['orba-settings']['bootstrap_mask']
    nr_random_masks = configuration['orba-settings']['nr_random_masks']
    keep_center = configuration['orba-settings']['keep_center']

    for subject in subjects:
        with torch.no_grad():
            averaged_mask = 0
            averaged_recon = 0
            for i in range(nr_random_masks):
                bootstrap_mask = generate_masks(
                    random_mask,
                    [data_dictionary['img_motion'][subject].shape[0], 1, 92, 1]
                )[:, 0, :, 0]
                prediction, _, _ = perform_reconstruction(
                    torch.tensor(data_dictionary['img_motion'][subject],
                                 dtype=torch.complex64, device=device),
                    torch.tensor(data_dictionary['sens_maps'][subject],
                                 dtype=torch.complex64, device=device),
                    torch.tensor(bootstrap_mask, dtype=torch.float32, device=device),
                    recon_model,
                    keep_central_point=keep_center
                )
                averaged_mask += 1/nr_random_masks * detach_torch(bootstrap_mask)
                averaged_recon += 1/nr_random_masks * abs(detach_torch(
                    prediction
                ))
            data_dictionary['img_orba'][subject] = averaged_recon
            data_dictionary['mask_orba'][subject] = averaged_mask

    return data_dictionary


def perform_sld_reconstructions(data_dictionary, configuration, subjects,
                                device="cuda"):
    """
    Perform reconstructions with the masks predicted by Supervised Line
    Detection.
    """

    data_dictionary['img_sld'] = {}
    data_dictionary['mask_sld'] = {}
    data_dictionary['mask_sld_thr'] = {}

    recon_model = load_recon_model(
        configuration['sld-settings']['recon_model'], device=device
    )
    keep_center = configuration['sld-settings']['keep_center']

    pretrained_weights = glob.glob(
        configuration['sld-settings']['sld_model']['weights']
    )[0]
    sld_model = load_model(
        pretrained_weights,
        configuration['sld-settings']['sld_model'],
        device=device
    )

    for subject in subjects:
        with torch.no_grad():
            data = torch.tensor(data_dictionary['img_motion'][subject],
                                dtype=torch.complex64, device=device)
            sens_maps = torch.tensor(data_dictionary['sens_maps'][subject],
                                     dtype=torch.complex64, device=device)
            kspace = process_kspace_linedet(
                data, sens_maps,
                coils_channel_dim=configuration['sld-settings']['coils_channel_dim'],
                coil_combined=configuration['sld-settings']['coil_combined']
            )

            predicted_mask = sld_model(kspace)

            # threshold at 0.5:
            predicted_mask_thr = torch.where(predicted_mask > 0.5,
                                             torch.tensor(1, device=device),
                                             torch.tensor(0, device=device))

            data_dictionary['mask_sld'][subject] = detach_torch(predicted_mask)
            data_dictionary['mask_sld_thr'][subject] = detach_torch(predicted_mask_thr)

            prediction, _, _ = perform_reconstruction(
                data, sens_maps, predicted_mask_thr.to(torch.float32), recon_model,
                keep_central_point=keep_center
            )

            data_dictionary['img_sld'][subject] = detach_torch(prediction)
            del data, sens_maps, kspace, predicted_mask, prediction, predicted_mask_thr
            torch.cuda.empty_cache()

    return data_dictionary


def save_metrics_to_hdf5(metrics, filename):
    with h5py.File(filename, "w") as h5file:
        for m_key, i_keys in metrics.items():
            m_group = h5file.create_group(m_key)
            for i_key, subjects in i_keys.items():
                i_group = m_group.create_group(i_key)
                for subject, array in subjects.items():
                    i_group.create_dataset(subject, data=array)


def load_metrics_from_hdf5(filename):
    metrics = {}
    with h5py.File(filename, "r") as h5file:
        for m_key, m_group in h5file.items():
            metrics[m_key] = {}
            for i_key, i_group in m_group.items():
                metrics[m_key][i_key] = {}
                for subject, dataset in i_group.items():
                    metrics[m_key][i_key][subject] = np.array(dataset)
    return metrics


def get_nii_header_from_h5(filename):
    filename_nii = os.path.dirname(os.path.realpath(str(filename)))
    filename_nii = filename_nii.replace("converted_raw/", "bids_from_raw/input/sub-")
    filename_nii = glob.glob(filename_nii + "/t2star/**echo-01_task-still_acq-fullres_T2star.nii.gz")[0]

    return nib.load(filename_nii).header


def calculate_precision_recall_threshold(pred, gt, threshold=0.5):
    # pred = 1 - pred
    # gt = 1 - gt
    true_positive = np.sum(np.logical_and(pred < threshold, gt == 0), axis=1)
    false_positive = np.sum(np.logical_and(pred < threshold, gt == 1), axis=1)
    false_negative = np.sum(np.logical_and(pred >= threshold, gt == 0), axis=1)

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)

    return precision, recall


def calculate_line_detection_metrics(data_dict, subject, mask_type, exp_name=None):
    metrics = {}
    if mask_type == "mask_phimo":
        data = data_dict[mask_type][exp_name][subject]
    else:
        data = data_dict[mask_type][subject]

    data_gt = data_dict["mask_gt"][subject]

    if len(data_dict["mask_gt"][subject]) > 0:
        metrics["correctly_excluded"] = (
            np.sum(np.logical_and(data < 0.5, data_gt == 0), axis=1)
            / np.sum(data_gt == 0, axis=1)
        )
        metrics["wrongly_excluded"] = (
            np.sum(np.logical_and(data < 0.5, data_gt == 1), axis=1)
            / np.sum(data_gt == 1, axis=1)
        )
        metrics["mae_masks"] = (np.mean(np.abs(data - data_gt), axis=1))
        metrics["accuracy"] = (
            np.sum(np.logical_and(data > 0.5, data_gt == 1), axis=1)
            + np.sum(np.logical_and(data < 0.5, data_gt == 0), axis=1)
        ) / data_gt.shape[1]

        metrics["accuracy_lowfreq"] = np.mean(
            np.sum(np.logical_and(data[:, 31:61] > 0.5, data_gt[:, 31:61] == 1),
                   axis=1)
            + np.sum(np.logical_and(data[:, 31:61] < 0.5, data_gt[:, 31:61] == 0),
                     axis=1)
        ) / data_gt[:, 31:61].shape[1]
        metrics["accuracy_medfreq"] = np.mean(
            np.sum(np.logical_and(data[:, np.r_[16:31, 61:76]] > 0.5,
                                  data_gt[:, np.r_[16:31, 61:76]] == 1),
                   axis=1)
            + np.sum(np.logical_and(data[:, np.r_[16:31, 61:76]] < 0.5,
                                    data_gt[:, np.r_[16:31, 61:76]] == 0),
                     axis=1)
        ) / data_gt[:, np.r_[16:31, 61:76]].shape[1]
        metrics["accuracy_highfreq"] = np.mean(
            np.sum(np.logical_and(data[:, np.r_[0:16, 76:92]] > 0.5,
                                  data_gt[:, np.r_[0:16, 76:92]] == 1),
                   axis=1)
            + np.sum(np.logical_and(data[:, np.r_[0:16, 76:92]] < 0.5,
                                    data_gt[:, np.r_[0:16, 76:92]] == 0),
                     axis=1)
        ) / data_gt[:, np.r_[0:16, 76:92]].shape[1]

        start_idx = 3*data_gt.shape[1] // 8
        end_idx = 5 * data_gt.shape[1] // 8
        (metrics["precision_central"],
         metrics["recall_central"]) = calculate_precision_recall_threshold(
            data[:, start_idx:end_idx], data_gt[:, start_idx:end_idx]
        )
        (metrics["precision_peripheral"],
         metrics["recall_peripheral"]) = calculate_precision_recall_threshold(
            np.hstack((data[:, :start_idx], data[:, end_idx:])),
            np.hstack((data_gt[:, :start_idx], data_gt[:, end_idx:]))
        )

    return metrics


def update_line_detection_metrics(line_detection_metrics, data_dict, subjects,
                                  mask_type, exp_names=None):
    if mask_type == "mask_phimo":
        for exp_name in exp_names:
            for subject in subjects:
                metrics = calculate_line_detection_metrics(data_dict, subject, mask_type, exp_name)
                for key, value in metrics.items():
                    if key in line_detection_metrics:
                        line_detection_metrics[key][exp_name][subject] = value
    else:
        for subject in subjects:
            metrics = calculate_line_detection_metrics(data_dict, subject, mask_type)
            for key, value in metrics.items():
                if key in line_detection_metrics:
                    line_detection_metrics[key][mask_type.replace("mask", "img")][subject] = value


def calculate_precision_recall(data_dict, subjects, mask_type, exp_name=None):

    # The precision-recall curve shows the tradeoff between precision and recall
    # for different thresholds. A high area under the curve represents both high
    # recall and high precision. High precision is achieved by having few false
    # positives in the returned results, and high recall is achieved by having
    # few false negatives in the relevant results. High scores for both show that
    # the classifier is returning accurate results (high precision), as well as
    # returning a majority of all relevant results (high recall).
    #
    # Precision is defined as the number of true positives over the number of
    # true positives plus the number of false positives.
    # Recall is defined as the number of true positives over the number of true
    # positives plus the number of false negatives.
    #
    # in our case: positive: 0, negative: 1

    metrics = {}
    if mask_type == "mask_phimo":
        data = [data_dict[mask_type][exp_name][subject] for subject in subjects]
    else:
        data = [data_dict[mask_type][subject] for subject in subjects]
    data_gt = [data_dict["mask_gt"][subject] for subject in subjects]

    data = np.concatenate(data).flatten()
    data_gt = np.concatenate(data_gt).flatten()

    data_inv = 1 - data
    data_gt_inv = 1 - data_gt

    positive_prevalence = np.sum(data_gt_inv == 1) / len(data_gt_inv)

    p, r, t = precision_recall_curve(data_gt_inv, data_inv)
    metrics["precision"] = p
    metrics["recall"] = r
    metrics["thresholds"] = t

    return metrics, positive_prevalence


def update_precision_recall(precision_recall, data_dict, subjects,
                            mask_type, exp_names=None):

    subjects_ = [sub for sub in subjects if len(data_dict["mask_gt"][sub]) > 0]

    if mask_type == "mask_phimo":
        for exp_name in exp_names:
            metrics, positive_prevalence = calculate_precision_recall(data_dict, subjects_, mask_type, exp_name)
            for key, value in metrics.items():
                precision_recall[key][exp_name] = value
    else:
        metrics, positive_prevalence = calculate_precision_recall(data_dict, subjects_, mask_type)
        for key, value in metrics.items():
            precision_recall[key][mask_type.replace("mask", "img")] = value

    return precision_recall, positive_prevalence


def calculate_deviations(data_dict, subjects, mask_type, exp_name=None):

    metrics = {}
    if mask_type == "mask_phimo":
        data = [data_dict[mask_type][exp_name][subject] for subject in subjects]
    else:
        data = [data_dict[mask_type][subject] for subject in subjects]
    data_gt = [data_dict["mask_gt"][subject] for subject in subjects]

    data = np.concatenate(data).flatten()
    data_gt = np.concatenate(data_gt).flatten()
    diff = data - data_gt

    metrics["motion_free"] = diff[data_gt == 1]
    metrics["motion_corrupted"] = diff[data_gt == 0]

    return metrics


def update_deviations(deviations, data_dict, subjects, mask_type,
                      exp_names=None):

    subjects_ = [sub for sub in subjects if len(data_dict["mask_gt"][sub]) > 0]

    if mask_type == "mask_phimo":
        for exp_name in exp_names:
            metrics = calculate_deviations(data_dict, subjects_, mask_type, exp_name)
            for key, value in metrics.items():
                deviations[key][exp_name] = value
    else:
        metrics = calculate_deviations(data_dict, subjects_, mask_type)
        for key, value in metrics.items():
            deviations[key][mask_type.replace("mask", "img")] = value

    return deviations


def calc_mask_stats(mask):
    """ Calculate the mean and percentage of excluded lines in the mask."""

    return np.mean(mask), np.sum(mask < 0.5) / mask.size * 100

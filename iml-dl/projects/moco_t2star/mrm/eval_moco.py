import os
import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from dl_utils.config_utils import set_seed
from data.t2star_loader import *
from projects.moco_t2star.utils import *
from mr_utils.parameter_fitting import T2StarFit, CalcSusceptibilityGradient
from projects.moco_t2star.mrm.utils import *
from projects.moco_t2star.mrm.utils_plot import *
from projects.moco_t2star.utils import detach_torch, statistical_testing, load_motion_data


set_seed(2109)

parser = argparse.ArgumentParser(description='Evaluate MoCo performance')
parser.add_argument('--config_path',
                    type=str,
                    default='./projects/moco_t2star/mrm/config_moco.yaml',
                    metavar='C',
                    help='path to configuration yaml file')
args = parser.parse_args()
with open(args.config_path, 'r') as stream_file:
    config = yaml.load(stream_file, Loader=yaml.FullLoader)
test_set = config["test_set"]
config = config["configurations"][test_set]
config["test_set"] = test_set
load_aal3 = True if config["test_set"] == "test" else False

if "set" in config.keys():
    subjects = np.loadtxt("./data/links_to_data/files_{}.txt".format(config["set"]),
                          dtype=str)
else:
    subjects = list(config["subjects"].keys())


if config["recalculate"]:
    if config["reregister"]:
        """ 1. Load the motion-corrupted data: """
        keys = ["sens_maps", "img_motion", 'img_motion_free', "img_hrqr",
                "brain_mask", "gray_matter", "white_matter", "subregions",
                "mask_gt", "slices_ind", "filename_motion_free"]
        data_dict = {key: {sub: [] for sub in subjects} for key in keys}

        for subject in subjects:
            data_dict = load_motion_data(subject, config, data_dict,
                                         load_aal3=load_aal3)

        for key in data_dict.keys():
            for subject in data_dict[key].keys():
                data_dict[key][subject] = np.array(data_dict[key][subject])

        device = ("cuda" if torch.cuda.is_available() else "cpu")
        data_dict = load_masks_into_data_dict(data_dict, config, subjects,
                                              device=device)
        gpu_memory_cleaning(log=False)


        """ 2. Perform PHIMO reconstructions with the predicted masks: """
        data_dict = perform_phimo_reconstructions(data_dict, config, subjects,
                                                  device=device)
        gpu_memory_cleaning(log=False)


        """ 3. Perfrom OR-BA and SLD reconstructions: """
        data_dict = perform_orba_reconstructions(data_dict, config, subjects,
                                                 device=device)
        gpu_memory_cleaning(log=False)
        data_dict = perform_sld_reconstructions(data_dict, config, subjects,
                                                device=device)
        gpu_memory_cleaning(log=False)


        """ 3. Calculate susceptibility gradients and T2* maps: """
        susc_gradients = {key: {} for key in ["img_motion", "img_motion_free"]}
        for subject in subjects:
            for key in ["img_motion", "img_motion_free"]:
                susc_gradients[key][subject] = []
                calc_susc_gradient = CalcSusceptibilityGradient(
                    complex_img=torch.tensor(np.rollaxis(data_dict[key][subject],
                                                         1, 4), dtype=torch.complex64),
                    mask=torch.tensor(data_dict["brain_mask"][subject], dtype=torch.float32),
                    mode="inference"
                )
                susc_grad_x, susc_grad_y, susc_grad_z = calc_susc_gradient()
                susc_gradients[key][subject] = np.sqrt(
                    detach_torch(susc_grad_x) ** 2
                    + detach_torch(susc_grad_y) ** 2
                    + detach_torch(susc_grad_z) ** 2
                )

        calc_t2star = T2StarFit(dim=4)
        t2star_maps = {key: {} for key in (["img_motion", "img_motion_free",
                                            "img_hrqr", "img_orba", "img_sld"]
                                           + list(config["experiments"]))}
        for subject in subjects:
            for key in ["img_motion", "img_motion_free", "img_hrqr", "img_orba",
                        "img_sld"]:
                tmp = calc_t2star(
                    torch.tensor(data_dict[key][subject], dtype=torch.complex64),
                    mask=None
                )
                t2star_maps[key][subject] = detach_torch(tmp)
            for exp_name in config["experiments"]:
                tmp = calc_t2star(
                    torch.tensor(data_dict["img_phimo"][exp_name][subject],
                                 dtype=torch.complex64),
                    mask=None
                )
                t2star_maps[exp_name][subject] = detach_torch(tmp)


        """ 4. Register to the motion-free data: """
        t2star_maps_reg = {key: {} for key in (
                ["img_motion", "img_motion_free", "img_hrqr", "img_orba",
                 "img_sld"]
                + list(config["experiments"])
        )}

        t2star_maps_reg_ants = {key: {} for key in (
                ["img_motion", "img_motion_free", "img_hrqr", "img_orba",
                 "img_sld"]
                + list(config["experiments"])
        )}

        for subject in subjects:
            header_motion_free = get_nii_header_from_h5(data_dict["filename_motion_free"][subject])

            filled_bm = []
            for i in range(0, data_dict["brain_mask"][subject].shape[0]):
                filled_bm.append(binary_fill_holes(data_dict["brain_mask"][subject][i]))
            filled_bm = np.array(filled_bm)

            t2star_maps["img_motion_free"][subject] = t2star_maps["img_motion_free"][subject] * filled_bm
            t2star_maps_reg["img_motion_free"][subject] = t2star_maps["img_motion_free"][subject]

            for i_key in (["img_motion", "img_hrqr", "img_orba", "img_sld"]
                          + list(config["experiments"])):
                if i_key in data_dict.keys():
                    moving_img = data_dict[i_key][subject]
                else:
                    moving_img = data_dict["img_phimo"][i_key][subject]

                t2star_reg, bm_invreg = reg_data_to_ref_spm(
                    img_ref=abs(data_dict["img_motion_free"][subject])[:, 0],
                    img=abs(moving_img)[:, 0],
                    other_images=[t2star_maps[i_key][subject]],
                    img_inv=data_dict["brain_mask"][subject],
                    header=header_motion_free,
                    tmp_dir=config["out_folder"] + "tmp_reg/"
                )
                t2star_reg *= filled_bm
                # clip the registered maps again to the range [0, 200]:
                t2star_reg = np.clip(t2star_reg, 0, 200)
                t2star_maps_reg[i_key][subject] = t2star_reg

                # mask the unregistered data with the inverse of the brain mask:
                bm_invreg = np.where(bm_invreg > 0.5, 1, 0)
                filled_bm_invreg = []
                for i in range(0, bm_invreg.shape[0]):
                    filled_bm_invreg.append(binary_fill_holes(bm_invreg[i]))
                filled_bm_invreg = np.array(filled_bm_invreg)
                t2star_maps[i_key][subject] = t2star_maps[i_key][subject] * filled_bm_invreg

        # replace nan values in t2star_maps_reg_spm by 0:
        for i_key in t2star_maps_reg.keys():
            for subject in subjects:
                t2star_maps_reg[i_key][subject] = np.nan_to_num(t2star_maps_reg[i_key][subject])

        # save intermediate results:
        out_dir = f"{config['out_folder']}registered_data/{config['test_set']}/"
        out_dir = check_folder(out_dir)

        with h5py.File(f"{out_dir}/t2star_maps_reg.h5", "w") as f:
            save_dict_to_hdf5(f, t2star_maps_reg)
        with h5py.File(f"{out_dir}/t2star_maps.h5", "w") as f:
            save_dict_to_hdf5(f, t2star_maps)

        for key in ["filename_motion_free",
                    "img_hrqr", "img_phimo", "img_orba",
                    "img_sld", "configs_train"]:
            data_dict.pop(key)
        with h5py.File(f"{out_dir}/data_dict.h5", "w") as f:
            save_dict_to_hdf5(f, data_dict)
        with h5py.File(f"{out_dir}/susc_gradients.h5", "w") as f:
            save_dict_to_hdf5(f, susc_gradients)

    else:
        # load the registered data:
        out_dir = f"{config['out_folder']}registered_data/{config['test_set']}/"
        with h5py.File(f"{out_dir}/t2star_maps_reg.h5", 'r') as f:
            t2star_maps_reg = load_dict_from_hdf5(f)
        with h5py.File(f"{out_dir}/t2star_maps.h5", 'r') as f:
            t2star_maps = load_dict_from_hdf5(f)
        with h5py.File(f"{out_dir}/data_dict.h5", 'r') as f:
            data_dict = load_dict_from_hdf5(f)
        with h5py.File(f"{out_dir}/susc_gradients.h5", 'r') as f:
            susc_gradients = load_dict_from_hdf5(f)

    # check for cut-off slices and exclude them from the evaluation:
    cutoff_slices = {}
    for subject in subjects:
        cutoff_slices[subject] = []
        for i_key in (["img_motion", "img_hrqr", "img_orba", "img_sld"]
                      + list(config["experiments"])):
            tmp = np.where(
                np.sum(
                    (t2star_maps_reg[i_key][subject] == 0) &
                    (data_dict["brain_mask"][subject] != 0),
                    axis=(1, 2)
                ) > 50,
            )[0]
            if len(tmp) > 0:
                cutoff_slices[subject].append(tmp)
        if len(cutoff_slices[subject]) > 0:
            cutoff_slices[subject] = np.unique(np.concatenate(cutoff_slices[subject]))

    # delete the cutoff slices from the registered t2* maps:
    data_dict["brain_mask_cut"] = {}
    data_dict["gray_matter_cut"] = {}
    data_dict["white_matter_cut"] = {}
    data_dict["subregions_cut"] = {}
    data_dict["slice_ind_cut"] = {}
    susc_gradients["img_motion_free_cut"] = {}
    for subject in subjects:
        for i_key in t2star_maps_reg.keys():
            t2star_maps_reg[i_key][subject] = np.delete(
                t2star_maps_reg[i_key][subject],
                cutoff_slices[subject],
                axis=0)
        data_dict["brain_mask_cut"][subject] = np.delete(
            np.copy(data_dict["brain_mask"][subject]),
            cutoff_slices[subject],
            axis=0
        )
        data_dict["gray_matter_cut"][subject] = np.delete(
            np.copy(data_dict["gray_matter"][subject]),
            cutoff_slices[subject],
            axis=0
        )
        data_dict["white_matter_cut"][subject] = np.delete(
            np.copy(data_dict["white_matter"][subject]),
            cutoff_slices[subject],
            axis=0
        )
        if load_aal3:
            data_dict["subregions_cut"][subject] = np.delete(
                np.copy(data_dict["subregions"][subject]),
                cutoff_slices[subject],
                axis=0
            )
        data_dict["slice_ind_cut"][subject] = np.delete(
            np.copy(data_dict["slices_ind"][subject]),
            cutoff_slices[subject]
        )
        susc_gradients["img_motion_free_cut"][subject] = np.delete(
            np.copy(susc_gradients["img_motion_free"][subject]),
            cutoff_slices[subject],
            axis=0
        )
        print("Due to registration, {} slices were cut off for subject {}.".format(
            len(cutoff_slices[subject]), subject
        ))


    """ 5. Evaluate quality of T2* maps: """
    # combine gray and white matter masks with condition that susceptibility
    # gradients are below threshold of 100 microT/m
    susc_threshold = 100
    mask_susc = {
        subject: np.logical_and(
            np.logical_or(data_dict["gray_matter_cut"][subject],
                          data_dict["white_matter_cut"][subject]),
            # data_dict["brain_mask_cut"][subject],
            susc_gradients["img_motion_free_cut"][subject] < susc_threshold
        ) for subject in subjects
    }

    metric_keys = {
        "mae": calc_masked_MAE,
        "ssim": calc_masked_SSIM_3D,
        "fsim": calc_masked_FSIM_3D,
        "lpips": calc_masked_LPIPS_3D
    }
    metrics = {m_key: {
        i_key: {} for i_key in t2star_maps.keys() if i_key != "img_motion_free"
    } for m_key in metric_keys}

    for subject in subjects:
        for m_key in metric_keys.keys():
            for i_key in metrics[m_key].keys():
                metrics[m_key][i_key][subject] = metric_keys[m_key](
                    t2star_maps_reg[i_key][subject],
                    t2star_maps_reg["img_motion_free"][subject],
                    mask_susc[subject]
                )

    # save the calculated metrics as a dictionary:
    out_file = "{}/metrics_{}_{}.h5".format(config["out_folder"],
                                           config["test_set"], config["tag"])
    save_metrics_to_hdf5(metrics, out_file)

else:
    # load previously calculated metrics with:
    out_file = "{}/metrics_{}_{}.h5".format(config["out_folder"],
                                              config["test_set"], config["tag"])
    metrics = load_metrics_from_hdf5(out_file)


img_keys = ['img_motion', 'img_orba', 'img_sld', 'AllSlices-NoKeepCenter',
            'Proposed', 'img_hrqr']

# merge the metrics, but divide by subject type (stronger, minor)
metric_keys =["mae", "ssim", "fsim", "lpips"]
metrics_merged = {m_key: {
    "stronger": {},
    "minor": {}
} for m_key in metric_keys}
worst_values = {"mae": np.amax, "ssim": np.amin, "fsim": np.amin, "lpips": np.amax}
for m_key in metric_keys:
    for i_key in img_keys:
        metrics_merged[m_key]["stronger"][i_key] = np.concatenate([
            metrics[m_key][i_key][subject]
            for subject in subjects
            if config["subjects"][subject] == "stronger"
        ])
        if "minor" in config["subjects"].values():
            metrics_merged[m_key]["minor"][i_key] = np.concatenate([
                metrics[m_key][i_key][subject]
                for subject in subjects
                if config["subjects"][subject] == "minor"
            ])

statistical_tests = {m_key: {
    "stronger": {},
    "minor": {}
} for m_key in metric_keys}

for m_key in metric_keys:
    for type in ["stronger", "minor"]:
        if type in config["subjects"].values():
            print(f"Metric: {m_key}, Type: {type}")
            comb, p = statistical_testing(
                img_keys,
                metrics_merged[m_key][type]
            )
            statistical_tests[m_key][type] = {
                "comb": comb,
                "p": p
            }

if config["test_set"] == "test":
    out_path = f"{config['out_folder']}/images_for_figures/image_quality_metrics/"
    os.makedirs(out_path, exist_ok=True)
    plot_violin_iq_metrics(metrics_merged, img_keys,
                           statistical_tests, p_value_threshold=0.001,
                           save_individual_plots=True, save_path=out_path)

    # look at mean values for the mask in the k-space center:
    mean_center = {exp_name: [] for exp_name in ["Proposed", "SLD",
                                                 "AllSlices-NoKeepCenter"]}
    for subject in subjects:
        if config["subjects"][subject] == "minor":
            mean_center["AllSlices-NoKeepCenter"].append(np.mean(data_dict["mask_phimo"]["AllSlices-NoKeepCenter"][subject][:, 41:51]))
            mean_center["Proposed"].append(np.mean(data_dict["mask_phimo"]["Proposed"][subject][:, 41:51]))
            mean_center["SLD"].append(np.mean(data_dict["mask_sld_thr"][subject][:, 41:51]))
    print("Mean values for the predicted masks in the k-space center:")
    for key in mean_center.keys():
        print(key, np.mean(mean_center[key]))



""" 6. Evaluate the line detection performance: """
keys = list(config["experiments"])+["img_orba", "img_sld"]
line_detection_metrics = {
    "correctly_excluded": {exp_name: {} for exp_name in keys},
    "wrongly_excluded": {exp_name: {} for exp_name in keys},
    "mae_masks": {exp_name: {} for exp_name in keys},
    "accuracy": {exp_name: {} for exp_name in keys},
    "precision_central": {exp_name: {} for exp_name in keys},
    "recall_central": {exp_name: {} for exp_name in keys},
    "precision_peripheral": {exp_name: {} for exp_name in keys},
    "recall_peripheral": {exp_name: {} for exp_name in keys},
}

# Update line detection metrics for both mask_phimo and mask_orba
update_line_detection_metrics(line_detection_metrics, data_dict,
                              subjects, "mask_phimo",
                              exp_names=config["experiments"])
update_line_detection_metrics(line_detection_metrics, data_dict, subjects,
                              "mask_orba")
update_line_detection_metrics(line_detection_metrics, data_dict, subjects,
                              "mask_sld")

precision_recall = {
    "precision": {exp_name: {} for exp_name in keys},
    "recall": {exp_name: {} for exp_name in keys},
    "thresholds": {exp_name: {} for exp_name in keys}
}
precision_recall, positive_prevalence = update_precision_recall(precision_recall, data_dict, subjects,
                                         "mask_phimo",
                                         exp_names=config["experiments"])
precision_recall, _ = update_precision_recall(precision_recall, data_dict, subjects,
                                         "mask_orba")
precision_recall, _ = update_precision_recall(precision_recall, data_dict, subjects,
                                           "mask_sld")

deviations = {
    "motion_free": {exp_name: {} for exp_name in keys},
    "motion_corrupted": {exp_name: {} for exp_name in keys}
}
deviations = update_deviations(deviations, data_dict, subjects, "mask_phimo",
                               exp_names=config["experiments"])
deviations = update_deviations(deviations, data_dict, subjects, "mask_orba")
deviations = update_deviations(deviations, data_dict, subjects, "mask_sld")


# Look at line detection metric values
if config["test_set"] == "test":
    # only choose subjects with ground truth masks:
    exps = ["img_orba", "img_sld", "AllSlices-NoKeepCenter", "Proposed"]
    subjects_ = [subject for subject in subjects if len(data_dict["mask_gt"][subject]) > 0]

    if len(subjects_) > 0:
        merged_line_det_metrics = {m_key: {} for m_key in [
            "accuracy", "mae_masks", "correctly_excluded", "precision_central",
            "recall_central", "precision_peripheral", "recall_peripheral"
        ]}
        for m_key in ["accuracy", "mae_masks", "correctly_excluded",
                      "precision_central", "recall_central",
                      "precision_peripheral", "recall_peripheral"]:
            for exp_name in line_detection_metrics[m_key].keys():
                merged_line_det_metrics[m_key][exp_name] = np.concatenate([
                    line_detection_metrics[m_key][exp_name][subject]
                    for subject in subjects_
                ])

        statistical_tests_ld = {m_key: {} for m_key in [
            "accuracy", "mae_masks", "correctly_excluded", "precision_central",
            "recall_central", "precision_peripheral", "recall_peripheral"
        ]}
        for m_key in ["accuracy", "mae_masks", "correctly_excluded",
                      "precision_central", "recall_central",
                      "precision_peripheral", "recall_peripheral"]:
            print(f"Metric: {m_key}")
            comb, p = statistical_testing(
                exps,
                merged_line_det_metrics[m_key]
            )
            statistical_tests_ld[m_key] = {
                "comb": comb,
                "p": p
            }

        out_path = f"{config['out_folder']}/images_for_figures/line_det_metrics/"
        os.makedirs(out_path, exist_ok=True)

        plot_violin_line_det_metrics(line_detection_metrics,
                                     exps, subjects_, statistical_tests_ld,
                                     metric_type="accuracy",
                                     p_value_threshold=0.001,
                                     save_path=out_path + "accuracy.png")
        plot_violin_line_det_metrics(line_detection_metrics,
                                     exps, subjects_, statistical_tests_ld,
                                     metric_type="mae_masks",
                                     p_value_threshold=0.001,
                                     save_path=out_path + "mae_masks.png")
        plot_pr_curves(precision_recall, exps,
                       save_path=out_path + "pr_curves.png",
                       positive_prevalence=positive_prevalence)
        # plot_precision_recall_kspace_loc(line_detection_metrics, exps, subjects_,
        #                                  save_path=out_path + "XXX_kspace_loc.png",
        #                                  statistical_tests=statistical_tests_ld,
        #                                  p_value_threshold=0.001)
        # plot_violin_line_det_metrics(line_detection_metrics,
        #                              exps, subjects_, statistical_tests_ld,
        #                              metric_type="correctly_excluded",
        #                              p_value_threshold=0.001,
        #                              save_path=out_path + "corr_excluded.png")
        # plot_class_differences(deviations, exps,
        #                        save_path=out_path + "class_differences_abs.png",
        #                        mode="abs")
        # plot_class_differences(deviations, exps,
        #                        save_path=out_path + "class_differences_rel.png",
        #                        mode="rel")


""" 7. Evaluate the image quality qualitatively: """
subjects_ = subjects

if config["test_set"] == "test-patterns":
    subject = "SQ-struct-04-p4-high"

    for slice_ind in [4, 14, 24]:
        ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]
        outfolder = f"{config['out_folder']}/images_for_figures/line_det/"
        os.makedirs(outfolder, exist_ok=True)
        individual_imshow(data_dict["mask_gt"][subject][ind],
                          save_path=f"{outfolder}mask_gt_{subject}_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_phimo"]["Proposed"][subject][ind],
                          save_path=f"{outfolder}mask_phimo_Proposed_{subject}"
                                    f"_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_phimo"]["AllSlices"][subject][ind],
                          save_path=f"{outfolder}mask_phimo_AllSlices_{subject}"
                                    f"_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_orba"][subject][ind],
                          save_path=f"{outfolder}mask_orba_{subject}"
                                    f"_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_sld_thr"][subject][ind],
                          save_path=f"{outfolder}mask_sld_{subject}"
                                    f"_slice_{slice_ind}.png")

    slice_ind = 17
    for subject in ["SQ-struct-04-p3-low", "SQ-struct-04-p3-mid", "SQ-struct-04-p3-high"]:
        outfolder = f"{config['out_folder']}/images_for_figures/example_images_patterns/"
        os.makedirs(outfolder, exist_ok=True)

        ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]
        cut_ind = np.where(data_dict["slice_ind_cut"][subject] == slice_ind)[0][0]
        data_shape = np.array(
            t2star_maps["img_motion_free"][subject][ind].T.shape)
        cut_img = 10

        individual_imshow(
            t2star_maps["img_motion_free"][subject][ind].T[:, cut_img // 2:-cut_img // 2],
            vmin=0, vmax=150, replace_nan=True,
            save_path=f"{outfolder}t2star_gt_{subject}_slice_{slice_ind}.png",
            text_left="MAE / SSIM:", fontsize=16
        )

        data_shape[1] -= cut_img
        fig, axs = plt.subplots(1, 4, figsize=(
        4 * 2, 2 * data_shape[0] / data_shape[1]))
        for descr, save, ax in zip(["img_motion", "AllSlices-NoKeepCenter",
                                    "Proposed", "img_hrqr"],
                                   ["motion", "AllSlices-NoKeepCenter",
                                    "Proposed", "hrqr"],
                                   axs):
            mae = metrics["mae"][descr][subject][ind]
            ssim = metrics["ssim"][descr][subject][ind]
            text = f"{mae:.1f} / {ssim:.2f}"
            add_imshow_axis(ax, t2star_maps[descr][subject][ind].T[:,
                                cut_img // 2:-cut_img // 2],
                            vmin=0, vmax=150, replace_nan=True,
                            text_mid=text, fontsize=16)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1,
                            bottom=0)
        plt.savefig(
            f"{outfolder}combined_t2star_{subject}_slice_{slice_ind}.png",
            dpi=400)
        plt.show()

        individual_imshow(data_dict["mask_gt"][subject][ind],
                          save_path=f"{outfolder}mask_gt_{subject}_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_phimo"]["Proposed"][subject][ind],
                          save_path=f"{outfolder}mask_phimo_Proposed_{subject}"
                                    f"_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_phimo"]["AllSlices-NoKeepCenter"][subject][ind],
                          save_path=f"{outfolder}mask_phimo_AllSlices-NoKeepCenter_"
                                    f"{subject}_slice_{slice_ind}.png")


if config["test_set"] == "test":
    for subject, slice_ind in zip(["SQ-struct-00", "SQ-struct-33"], [15, 15]):
        outfolder = f"{config['out_folder']}/images_for_figures/example_images//"
        os.makedirs(outfolder, exist_ok=True)
        ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]
        data_shape = np.array(
            t2star_maps["img_motion_free"][subject][ind].T.shape)
        cut_img = 10
        data_shape[1] -= cut_img
        fig, axs = plt.subplots(1, 6, figsize=(
        6 * 2, 2 * data_shape[0] / data_shape[1]))
        for descr, save, ax in zip(["img_motion", "img_orba", "img_sld",
                                    "Proposed", "img_hrqr"],
                                   ["motion", "orba", "sld", "Proposed",
                                    "hrqr"],
                                   axs[:-1]):
            mae = metrics["mae"][descr][subject][ind]
            ssim = metrics["ssim"][descr][subject][ind]
            text = f"{mae:.1f} / {ssim:.2f}"
            add_imshow_axis(ax,
                            t2star_maps[descr][subject][ind].T[:, cut_img // 2:-cut_img // 2],
                            vmin=0, vmax=150, replace_nan=True,
                            text_mid=text, fontsize=16)
        add_imshow_axis(axs[-1],
                        t2star_maps["img_motion_free"][subject][ind].T[:,
                        cut_img // 2:-cut_img // 2],
                        vmin=0, vmax=150, replace_nan=True,
                        text_mid="MAE / SSIM", fontsize=16)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1,
                            bottom=0)
        plt.savefig(
            f"{outfolder}combined_t2star_{subject}_slice_{slice_ind}.png",
            dpi=400)
        plt.show()

        if data_dict["mask_gt"][subject].shape[0] > 0:
            individual_imshow(data_dict["mask_gt"][subject][ind],
                              save_path=f"{outfolder}mask_gt_{subject}_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_phimo"]["Proposed"][subject][ind],
                          save_path=f"{outfolder}mask_phimo_Proposed_{subject}"
                                    f"_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_orba"][subject][ind],
                          save_path=f"{outfolder}mask_orba_{subject}"
                                    f"_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_sld_thr"][subject][ind],
                          save_path=f"{outfolder}mask_sld_{subject}"
                                    f"_slice_{slice_ind}.png")



if config["test_set"] == "test-extreme":
    for subject, slice_ind in zip(["SQ-struct-02"], [15, 15]):
        outfolder = (f"{config['out_folder']}/images_for_figures/"
                     f"example_images_extreme/")
        os.makedirs(outfolder, exist_ok=True)
        ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]
        data_shape = np.array(
            t2star_maps["img_motion_free"][subject][ind].T.shape)
        cut_img = 10
        data_shape[1] -= cut_img
        fig, axs = plt.subplots(1, 6, figsize=(
        6 * 2, 2 * data_shape[0] / data_shape[1]))
        for descr, save, ax in zip(["img_motion", "img_orba", "img_sld",
                                    "Proposed", "img_hrqr"],
                                   ["motion", "orba", "sld", "Proposed",
                                    "hrqr"],
                                   axs[0:-1]):
            mae = metrics["mae"][descr][subject][ind]
            ssim = metrics["ssim"][descr][subject][ind]
            text = f"{mae:.1f} / {ssim:.2f}"
            add_imshow_axis(ax,
                            t2star_maps[descr][subject][ind].T[:, cut_img // 2:-cut_img // 2],
                            vmin=0, vmax=150, replace_nan=True,
                            text_mid=text, fontsize=16)
        add_imshow_axis(axs[-1],
                        t2star_maps["img_motion_free"][subject][ind].T[:,
                        cut_img // 2:-cut_img // 2],
                        vmin=0, vmax=150, replace_nan=True,
                        text_mid="MAE / SSIM", fontsize=16)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1,
                            bottom=0)
        plt.savefig(
            f"{outfolder}combined_t2star_{subject}_slice_{slice_ind}.png",
            dpi=400)
        plt.show()

        if data_dict["mask_gt"][subject].shape[0] > 0:
            individual_imshow(data_dict["mask_gt"][subject][ind],
                              save_path=f"{outfolder}mask_gt_{subject}_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_phimo"]["Proposed"][subject][ind],
                          save_path=f"{outfolder}mask_phimo_Proposed_{subject}"
                                    f"_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_orba"][subject][ind],
                          save_path=f"{outfolder}mask_orba_{subject}"
                                    f"_slice_{slice_ind}.png")
        individual_imshow(data_dict["mask_sld_thr"][subject][ind],
                          save_path=f"{outfolder}mask_sld_{subject}"
                                    f"_slice_{slice_ind}.png")

        print("Excluded lines in k-space center (GT mask):",
              np.count_nonzero(data_dict['mask_gt'][subject][ind][92//2-10:92//2+10]==0))


""" 9. Look at sub-region T2* accuracy: """
if load_aal3:
    img_keys = ['img_motion', 'img_orba', 'img_sld', 'AllSlices-NoKeepCenter',
                'Proposed', 'img_hrqr', 'img_motion_free']
    region_mean_t2star = {exp_name: {
        i: [] for i in range(1, 171)
    } for exp_name in img_keys}
    region_std_t2star = {exp_name: {
        i: [] for i in range(1, 171)
    } for exp_name in img_keys}


    for subregion_idx in range(1, 171):
        for subject in subjects:
            roi = data_dict["subregions_cut"][subject] == subregion_idx
            for exp_name in img_keys:
                region_mean_t2star[exp_name][subregion_idx].append(
                    t2star_maps_reg[exp_name][subject][roi].mean()
                )
                region_std_t2star[exp_name][subregion_idx].append(
                    t2star_maps_reg[exp_name][subject][roi].std()
                )

    color_dict = {
        "img_motion": ["#BEBEBE", 0.7],
        "img_orba": ["#8497B0", 0.8],
        "img_sld": ["#005293", 0.75],
        "img_hrqr": ["#44546A", 0.9],
        "AllSlices-NoKeepCenter": ["#A9C09A", 0.8],
        "Proposed": ["#6C8B57", 0.9]
    }

    keys_plot = [k for k in img_keys if k != "img_motion_free"]
    n_keys = len(keys_plot)
    fig, axes = plt.subplots(2, n_keys, figsize=(4 * n_keys, 6), sharey='row', sharex='all')

    for ax, key in zip(axes[0, :], keys_plot):
        data = []
        for i in range(1, 171):
            data_i = np.array(region_mean_t2star[key][i]) - np.array(region_mean_t2star["img_motion_free"][i])
            ax.scatter(region_mean_t2star["img_motion_free"][i],
                       data_i,
                       c=color_dict[key][0],
                       alpha=color_dict[key][1],
                       s=1)
            data.append(data_i)
        ax.axhline(np.nanmean(data), color='black', linestyle='--', linewidth=1)
        ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--')
        if ax == axes[0, 0]:
            ax.set_ylabel("Difference mean T2* [ms]", fontsize=16)

    for ax, key in zip(axes[1, :], keys_plot):
        data = []
        for i in range(1, 171):
            ax.scatter(region_mean_t2star["img_motion_free"][i],
                       np.array(region_std_t2star[key][i]),
                       c=color_dict[key][0],
                       alpha=color_dict[key][1],
                       s=1)
            data.append(region_std_t2star[key][i])
        ax.set_xlabel("Mean T2* [ms] \n(motion-free reference)", fontsize=16)
        ax.axhline(np.nanmean(data), color='black', linestyle='--', linewidth=1)
        ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--')
        if ax == axes[1, 0]:
            ax.set_ylabel("Std T2* [ms]", fontsize=16)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.9, wspace=0.1, hspace=0.2)
    plt.show()


print("Done")

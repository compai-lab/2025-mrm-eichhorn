import torch
import yaml
import argparse
import numpy as np
import gc
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
                    default='./projects/moco_t2star/mrm/config_moco_simulated.yaml',
                    metavar='C',
                    help='path to configuration yaml file')
args = parser.parse_args()
with open(args.config_path, 'r') as stream_file:
    config = yaml.load(stream_file, Loader=yaml.FullLoader)
test_set = config["test_set"]
config = config["configurations"][test_set]
config["test_set"] = test_set

if "set" in config.keys():
    subjects = np.loadtxt("./data/links_to_data/files_{}.txt".format(config["set"]),
                          dtype=str)
else:
    subjects = list(config["subjects"].keys())

if config["recalculate"]:
    """ 1. Load the motion-corrupted data: """
    keys = ["sens_maps", "img_motion", 'img_motion_free', "img_hrqr",
            "brain_mask", "gray_matter", "white_matter", "mask_gt",
            "slices_ind", "filename_motion_free"]
    data_dict = {key: {sub: [] for sub in subjects} for key in keys}

    for subject in subjects:
        data_dict = load_motion_data(subject, config, data_dict)

    for key in data_dict.keys():
        for subject in data_dict[key].keys():
            data_dict[key][subject] = np.array(data_dict[key][subject])

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    data_dict = load_masks_into_data_dict(data_dict, config, subjects, device=device)
    gpu_memory_cleaning(log=False)


    """ 2. Perform PHIMO reconstructions with the predicted masks: """
    data_dict = perform_phimo_reconstructions(data_dict, config, subjects, device=device)
    gpu_memory_cleaning(log=False)


    """ 3. Perform OR-BA and SLD reconstructions: """
    data_dict = perform_orba_reconstructions(data_dict, config, subjects,
                                             device=device)
    gpu_memory_cleaning(log=True)
    data_dict = perform_sld_reconstructions(data_dict, config, subjects,
                                            device=device)
    gpu_memory_cleaning(log=True)


    """ 4. Calculate susceptibility gradients and T2* maps: """
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
                                        "img_orba", "img_sld", "img_hrqr"]
                                       + list(config["experiments"]))}
    for subject in subjects:
        for key in ["img_motion", "img_motion_free", "img_orba", "img_sld",
                    "img_hrqr"]:
            tmp = calc_t2star(
                torch.tensor(data_dict[key][subject], dtype=torch.complex64),
                mask=None
            )
            t2star_maps[key][subject] = detach_torch(tmp)
        for exp_name in config["experiments"]:
            tmp = calc_t2star(
                torch.tensor(data_dict["img_phimo"][exp_name][subject], dtype=torch.complex64),
                mask=None
            )
            t2star_maps[exp_name][subject] = detach_torch(tmp)

    for subject in subjects:
        for i_key in t2star_maps.keys():
            # mask the (unregistered) data with the brain mask:
            bm = np.where(data_dict["brain_mask"][subject] > 0.5, 1, 0)
            filled_bm = []
            for i in range(0, bm.shape[0]):
                filled_bm.append(binary_fill_holes(bm[i]))
            filled_bm = np.array(filled_bm)
            t2star_maps[i_key][subject] = t2star_maps[i_key][subject] * filled_bm

    # save intermediate results:
    out_dir = f"{config['out_folder']}registered_data/{config['test_set']}/"
    out_dir = check_folder(out_dir)

    with h5py.File(f"{out_dir}/t2star_maps.h5", "w") as f:
        save_dict_to_hdf5(f, t2star_maps)

    for key in ["filename_motion_free", "sens_maps", "img_motion",
                "img_motion_free", "img_hrqr", "img_phimo", "img_orba",
                "img_sld", "configs_train"]:
        data_dict.pop(key)
    with h5py.File(f"{out_dir}/data_dict.h5", "w") as f:
        save_dict_to_hdf5(f, data_dict)
    with h5py.File(f"{out_dir}/susc_gradients.h5", "w") as f:
        save_dict_to_hdf5(f, susc_gradients)

else:
    # load the registered data:
    out_dir = f"{config['out_folder']}registered_data/{config['test_set']}/"
    with h5py.File(f"{out_dir}/t2star_maps.h5", 'r') as f:
        t2star_maps = load_dict_from_hdf5(f)
    with h5py.File(f"{out_dir}/data_dict.h5", 'r') as f:
        data_dict = load_dict_from_hdf5(f)
    with h5py.File(f"{out_dir}/susc_gradients.h5", 'r') as f:
        susc_gradients = load_dict_from_hdf5(f)


""" 5. Evaluate quality of T2* maps: """
# combine gray and white matter masks with condition that susceptibility
# gradients are below threshold of 100 microT/m
susc_threshold = 100
mask_susc = {
    subject: np.logical_and(
        np.logical_or(data_dict["gray_matter"][subject],
                      data_dict["white_matter"][subject]),
        susc_gradients["img_motion_free"][subject] < susc_threshold
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
                t2star_maps[i_key][subject],
                t2star_maps["img_motion_free"][subject],
                mask_susc[subject]
            )

# save the calculated metrics as a dictionary:
out_file = "{}/metrics_{}_{}.h5".format(config["out_folder"],
                                       config["test_set"], config["tag"])
save_metrics_to_hdf5(metrics, out_file)


img_keys = ["img_motion", "img_orba", "img_sld", "AllSlices-NoKeepCenter",
            "Proposed", "img_hrqr"]

# merge the metrics, but divide by subject type (stronger, minor)
metric_keys =["mae", "ssim", "fsim", "lpips"]
metrics_merged = {m_key: {
    "all_simulated": {}
} for m_key in metric_keys}

for m_key in metric_keys:
    for i_key in img_keys:
        metrics_merged[m_key]["all_simulated"][i_key] = np.concatenate([
            metrics[m_key][i_key][subject]
            for subject in subjects
        ])

statistical_tests = {m_key: {
    "all_simulated": {}
} for m_key in metric_keys}

for m_key in metric_keys:
    print(f"Metric: {m_key}")
    comb, p = statistical_testing(
        img_keys,
        metrics_merged[m_key]["all_simulated"]
    )
    statistical_tests[m_key]["all_simulated"] = {
        "comb": comb,
        "p": p
    }

out_path = (f"{config['out_folder']}/images_for_figures/"
            f"image_quality_metrics_simulated/")
os.makedirs(out_path, exist_ok=True)
ylims = {'mae': 32, 'ssim': None, 'fsim': None, 'lpips': None}
plot_violin_iq_metrics(metrics_merged, img_keys,
                       statistical_tests, p_value_threshold=0.001,
                       save_individual_plots=True, save_path=out_path,
                       ylims=ylims)

for i_key in img_keys:
    tmp = metrics_merged['mae']['all_simulated'][i_key]
    if np.amax(tmp) > 32:
        print(i_key, "- MAE values larger than 32:")
        print(tmp[tmp>32])



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


# Look at metric values
exps = ["img_orba", "img_sld", "AllSlices-NoKeepCenter", "Proposed"]

merged_line_det_metrics = {m_key: {} for m_key in [
    "accuracy", "mae_masks", "correctly_excluded", "precision_central",
    "recall_central", "precision_peripheral", "recall_peripheral"
]}

for m_key in ["accuracy", "mae_masks", "correctly_excluded", "precision_central",
              "recall_central", "precision_peripheral", "recall_peripheral"]:
    for exp_name in line_detection_metrics[m_key].keys():
        merged_line_det_metrics[m_key][exp_name] = np.concatenate([
            line_detection_metrics[m_key][exp_name][subject]
            for subject in subjects
        ])

statistical_tests_ld = {m_key: {} for m_key in [
    "accuracy", "mae_masks", "correctly_excluded", "precision_central",
    "recall_central", "precision_peripheral", "recall_peripheral"
]}

for m_key in ["accuracy", "mae_masks", "correctly_excluded", "precision_central",
              "recall_central", "precision_peripheral", "recall_peripheral"]:
    print(f"Metric: {m_key}")
    comb, p = statistical_testing(
        exps,
        merged_line_det_metrics[m_key]
    )
    statistical_tests_ld[m_key] = {
        "comb": comb,
        "p": p
    }

out_path = f"{config['out_folder']}/images_for_figures/line_det_metrics_simulated/"
os.makedirs(out_path, exist_ok=True)
plot_violin_line_det_metrics(line_detection_metrics,
                             exps, subjects, statistical_tests_ld,
                             metric_type="accuracy", p_value_threshold=0.001,
                             save_path=out_path+"accuracy.png")
plot_violin_line_det_metrics(line_detection_metrics,
                             exps, subjects, statistical_tests_ld,
                             metric_type="mae_masks", p_value_threshold=0.001,
                             save_path=out_path+"mae_masks.png")
plot_pr_curves(precision_recall, exps, save_path=out_path+"pr_curves.png",
               positive_prevalence=positive_prevalence)
# plot_precision_recall_kspace_loc(line_detection_metrics, exps, subjects,
#                                  save_path=out_path+"XXX_kspace_loc.png",
#                                  statistical_tests=statistical_tests_ld,
#                                  p_value_threshold=0.001)
# plot_class_differences(deviations, exps,
#                        save_path=out_path+"class_differences_abs.png", mode="abs")
# plot_class_differences(deviations, exps,
#                        save_path=out_path+"class_differences_rel.png",
#                        mode="rel")
# plot_violin_line_det_metrics(line_detection_metrics,
#                              exps, subjects, statistical_tests_ld,
#                              metric_type="correctly_excluded",
#                              p_value_threshold=0.001,
#                              save_path=out_path+"corr_excluded.png")


# look at distribution over slices:
plot_line_det_metrics(line_detection_metrics, data_dict["slices_ind"],
                      ['Proposed', 'AllSlices'],
                      subjects, "accuracy", split_by_level=False)



subject = "SQ-struct-44-sim00"

for slice_ind in [4, 14, 24]:
    ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]
    outfolder = f"{config['out_folder']}/images_for_figures/line_det_simulated/"
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


# Look for subject with good accuracy and bad acuracy:
print("Subjects with high accuracy:")
for subject in subjects:
    if np.mean(line_detection_metrics["accuracy"]["Proposed"][subject]) > 0.95:
        print(subject, np.mean(line_detection_metrics["accuracy"]["Proposed"][subject]))
print("Subjects with low accuracy:")
for subject in subjects:
    if np.mean(line_detection_metrics["accuracy"]["Proposed"][subject]) < 0.8:
        print(subject, np.mean(line_detection_metrics["accuracy"]["Proposed"][subject]))
print("#####")

slice_ind = 17
for subject, folder in zip(["SQ-struct-44-sim00", "SQ-struct-48-sim00"],
                           ["high_accuracy", "low_accuracy"]):
    outfolder = f"{config['out_folder']}/images_for_figures/example_images_simulated/{folder}/"
    os.makedirs(outfolder, exist_ok=True)
    ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]
    print(subject, line_detection_metrics["accuracy"]["Proposed"][subject][ind])
    data_shape = np.array(t2star_maps["img_motion_free"][subject][ind].T.shape)
    cut_right = 10
    data_shape[1] -= cut_right
    fig, axs = plt.subplots(1, 6, figsize=(6*2, 2*data_shape[0]/data_shape[1]))
    for descr, save, ax in zip(["img_motion", "img_orba", "img_sld",
                                "Proposed", "img_hrqr"],
                               ["motion", "orba", "sld", "Proposed", "hrqr"],
                               axs[0:-1]):
        mae = metrics["mae"][descr][subject][ind]
        ssim = metrics["ssim"][descr][subject][ind]
        text = f"{mae:.1f} / {ssim:.2f}"
        add_imshow_axis(ax, t2star_maps[descr][subject][ind].T[:, cut_right//2:-cut_right//2],
                        vmin=0, vmax=150, replace_nan=True,
                        text_mid=text, fontsize=16)
    add_imshow_axis(axs[-1], t2star_maps["img_motion_free"][subject][ind].T[:, cut_right//2:-cut_right//2],
                    vmin=0, vmax=150, replace_nan=True, text_mid="MAE / SSIM",
                    fontsize=16)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.savefig(f"{outfolder}combined_t2star_{subject}_slice_{slice_ind}.png", dpi=400)
    plt.show()

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



print("Done")

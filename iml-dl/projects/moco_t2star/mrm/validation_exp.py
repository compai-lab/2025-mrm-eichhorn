import os
import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dl_utils.config_utils import set_seed
from data.t2star_loader import *
from projects.moco_t2star.utils import *
from mr_utils.parameter_fitting import T2StarFit
from projects.moco_t2star.mrm.utils import *
from projects.moco_t2star.mrm.utils_plot import *
from projects.moco_t2star.utils import detach_torch
from optim.losses.physics_losses import ModelFitError

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
test_set = "test-patterns"
config = config["configurations"][test_set]
config["test_set"] = test_set
load_aal3 = False

os.makedirs(f"{config['out_folder']}/images_for_figures/validation/",
            exist_ok=True)

if "set" in config.keys():
    subjects = np.loadtxt("./data/links_to_data/files_{}.txt".format(config["set"]),
                          dtype=str)
else:
    subjects = list(config["subjects"].keys())


""" 1. Load data and calculate metrics: """
if config["recalculate"]:
    if config["reregister"]:
        print("ERROR: Please first run the eval_moco.py script for the"
              " registration step.")
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
            cutoff_slices[subject] = np.unique(
                np.concatenate(cutoff_slices[subject]))

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
        print(
            "Due to registration, {} slices were cut off for subject {}.".format(
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
        i_key: {} for i_key in t2star_maps.keys() if
        i_key != "img_motion_free"
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
                                            config["test_set"],
                                            config["tag"])
    save_metrics_to_hdf5(metrics, out_file)

else:
    # load previously calculated metrics with:
    out_file = "{}/metrics_{}_{}.h5".format(config["out_folder"],
                                            config["test_set"], config["tag"])
    metrics = load_metrics_from_hdf5(out_file)

keys = list(config["experiments"])+["img_orba", "img_sld"]
line_detection_metrics = {
    "correctly_excluded": {exp_name: {} for exp_name in keys},
    "wrongly_excluded": {exp_name: {} for exp_name in keys},
    "mae_masks": {exp_name: {} for exp_name in keys},
    "accuracy": {exp_name: {} for exp_name in keys},
    "accuracy_lowfreq": {exp_name: {} for exp_name in keys},
    "accuracy_medfreq": {exp_name: {} for exp_name in keys},
    "accuracy_highfreq": {exp_name: {} for exp_name in keys},
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


""" 2. Plot Line Detection Accuracy per Slice: """
plot_line_det_metrics(line_detection_metrics, data_dict["slices_ind"],
                      ['Proposed', 'AllSlices'],
                      subjects, "accuracy", split_by_level=False,
                      save_path=f"{config['out_folder']}/images_for_figures"
                                f"/validation/EvenOdd.png")

# example slices:
subject = "SQ-struct-04-p4-high"
for slice_ind in [4, 14, 24]:
    ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]
    outfolder = f"{config['out_folder']}/images_for_figures/validation/line_det/"
    os.makedirs(outfolder, exist_ok=True)
    individual_imshow(data_dict["mask_gt"][subject][ind],
                      save_path=f"{outfolder}mask_gt_{subject}_slice_{slice_ind}.png")
    individual_imshow(data_dict["mask_phimo"]["Proposed"][subject][ind],
                      save_path=f"{outfolder}mask_phimo_Proposed_{subject}"
                                f"_slice_{slice_ind}.png")
    individual_imshow(data_dict["mask_phimo"]["AllSlices"][subject][ind],
                      save_path=f"{outfolder}mask_phimo_AllSlices_{subject}"
                                f"_slice_{slice_ind}.png")

subject = "SQ-struct-04-p2-high"
for slice_ind in [3, 13, 23]:
    ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]
    outfolder = f"{config['out_folder']}/images_for_figures/validation/line_det/"
    os.makedirs(outfolder, exist_ok=True)
    individual_imshow(data_dict["mask_gt"][subject][ind],
                      save_path=f"{outfolder}mask_gt_{subject}_slice_{slice_ind}.png")
    individual_imshow(data_dict["mask_phimo"]["Proposed"][subject][ind],
                      save_path=f"{outfolder}mask_phimo_Proposed_{subject}"
                                f"_slice_{slice_ind}.png")
    individual_imshow(data_dict["mask_phimo"]["AllSlices"][subject][ind],
                      save_path=f"{outfolder}mask_phimo_AllSlices_{subject}"
                                f"_slice_{slice_ind}.png")


"""2a. Plot Line detection metrics for different k-space regions: """
plot_violin_line_det_freq_dep(line_detection_metrics,
                              ["AllSlices-NoKeepCenter", "Proposed"],
                              save_path=f"{config['out_folder']}/images_for_figures"
                              f"/validation/line_det_freq_dep.png")


""" 3. Example reconstructions for central four k-space points: """
subject = "SQ-struct-04-p4-low"
mask = np.zeros(92)
img_zf, _, _ = apply_undersampling_mask_torch(
    torch.tensor([data_dict['img_motion_free'][subject][10]],
                 dtype=torch.complex64),
    torch.tensor([data_dict['sens_maps'][subject][10]],
                 dtype=torch.complex64),
    torch.tensor([mask],
                 dtype=torch.float32),
    keep_central_point=True,
    add_mean_fs=False
)
img_zf_motion, _, _ = apply_undersampling_mask_torch(
    torch.tensor([data_dict['img_motion'][subject][10]],
                 dtype=torch.complex64),
    torch.tensor([data_dict['sens_maps'][subject][10]],
                 dtype=torch.complex64),
    torch.tensor([mask],
                 dtype=torch.float32),
    keep_central_point=True,
    add_mean_fs=False
)
grid_imshow_center_recons(detach_torch(img_zf), detach_torch(img_zf_motion),
                          echoes=[0, 3, 6, 9],
                          save_path=f"{config['out_folder']}/images_for_figures"
                                    f"/validation/kspace_center.png")


""" 4. Compare reconstruction quality with and without KeepCenter: """
device = "cuda" if torch.cuda.is_available() else "cpu"
recon_models = {}
for exp, recon_key in zip(["AllSlices-NoKeepCenter", "Proposed"],
                          ["NoKeepCenter", "KeepCenter"]):
    exp_id = config["experiments"][exp]["id"]
    config_path = glob.glob(f"{config['checkpoint_path']}weights/"
                            f"{exp_id}/**/**/config.yaml")[0]
    with open(config_path, 'r') as stream_file:
        config_exp = yaml.load(
            stream_file, Loader=yaml.FullLoader
        )
    recon_model = load_recon_model(
        config_exp['recon_model_downstream'],
        device=device
    )
    recon_models[recon_key] = recon_model

masks = {"mask_1": np.ones(92), "mask_2": np.ones(92), "mask_3": np.ones(92)}

masks["mask_1"][0:5] = 0
masks["mask_1"][20:23] = 0
masks["mask_1"][42:49] = 0
masks["mask_1"][82:88] = 0

masks["mask_2"][0:7] = 0
masks["mask_2"][20:23] = 0
masks["mask_2"][43:48] = 0
masks["mask_2"][82:88] = 0

masks["mask_3"][0:7] = 0
masks["mask_3"][20:23] = 0
masks["mask_3"][44:47] = 0
masks["mask_3"][82:90] = 0

reconstructions = {}
for mask_key in masks.keys():
    reconstructions[mask_key] = {}
    for recon_key in recon_models.keys():
        reconstructions[mask_key][recon_key] = {subject: [] for subject in config["subjects"]}
        reconstructions[mask_key]["zero-filled-"+recon_key] = {
            subject: [] for subject in subjects
        }

print("Exclusion rates: ")
for key in masks.keys():
    print(key, (1-np.count_nonzero(masks[key])/len(masks[key]))*100)

keep_center = {"KeepCenter": True, "NoKeepCenter": False}

for mask_key in masks.keys():
    for subject in subjects:
        for idx in range(0, len(data_dict["slices_ind"][subject])):
            for recon_key, recon_model in recon_models.items():
                prediction, img_zf, _ =perform_reconstruction(
                torch.tensor([data_dict['img_motion_free'][subject][idx]],
                             dtype=torch.complex64, device=device),
                torch.tensor([data_dict['sens_maps'][subject][idx]],
                             dtype=torch.complex64, device=device),
                torch.tensor([masks[mask_key]],
                             dtype=torch.float32, device=device),
                recon_model,
                keep_central_point=keep_center[recon_key]
                )
                reconstructions[mask_key][recon_key][subject].append(detach_torch(prediction[0]))
                reconstructions[mask_key]["zero-filled-"+recon_key][subject].append(detach_torch(img_zf[0]))

# plot example reconstructions:
subject = ("SQ-struct-04-p1-low" if "SSQ-struct-04-p1-low" in list(config["subjects"].keys())
           else list(config["subjects"].keys())[0])
slice_idx = 10
print("Showing slice ", data_dict["slices_ind"][subject][10])
echo = 0
data_shape = np.array(data_dict["img_motion_free"][subject][slice_idx][echo].T.shape)
cut_img = 0
data_shape[1] -= cut_img
figsize = (3 * 2, 2 * data_shape[0] / data_shape[1])
save_path = f"{config['out_folder']}/images_for_figures/validation/recon/"
os.makedirs(save_path, exist_ok=True)

for i, mask_key in enumerate(masks.keys()):
    individual_imshow(np.repeat(masks[mask_key][None], 112, axis=0),
                      save_path=save_path + f"{mask_key }.png")

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for recon_key, ax in zip(["NoKeepCenter", "KeepCenter", "fully-sampled"],
                             axs):
        if recon_key == "fully-sampled":
            img = data_dict["img_motion_free"][subject][slice_idx][echo].T
        else:
            img = reconstructions[mask_key][recon_key][subject][slice_idx][echo].T
        if cut_img > 0:
            add_imshow_axis(ax, abs(img[:, cut_img // 2:-cut_img // 2]))
        else:
            add_imshow_axis(ax, abs(img))
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1,
                        bottom=0)
    plt.savefig(
        f"{save_path}combined_recons_{mask_key}_{subject}_slice_{slice_ind}.png",
        dpi=400)
    plt.show()


central_lines = {"mask_1": "7 central lines",
                 "mask_2": "5 central lines",
                 "mask_3": "3 central lines"}
recon_labels = {"zero-filled": "Zero-filled",
                "fully-sampled": "Fully-sampled",
                "NoKeepCenter": "Original \nmask",
                "KeepCenter": "KeepCenter"}


t2star_maps = {
    mask_key: {
        recon_key: {
            subject: [] for subject in reconstructions[mask_key][recon_key].keys()
        } for recon_key in reconstructions[mask_key].keys()
    } for mask_key in reconstructions.keys()
}
calc_t2star = T2StarFit(dim=4)

for mask_key in masks.keys():
    for subject in config["subjects"]:
        for recon_key, recon_model in recon_models.items():
            tmp = calc_t2star(
                torch.tensor(reconstructions[mask_key][recon_key][subject]),
                mask=None
            )
            t2star_maps[mask_key][recon_key][subject] = detach_torch(tmp)

t2star_maps_fully_sampled = {subject: [] for subject in config["subjects"]}
for subject in config["subjects"]:
    tmp = calc_t2star(
        torch.tensor(data_dict["img_motion_free"][subject]),
        mask=None
    )
    t2star_maps_fully_sampled[subject] = detach_torch(tmp)

susc_threshold = 100
mask_susc = {
    subject: np.logical_and(
        np.logical_or(data_dict["gray_matter"][subject],
                      data_dict["white_matter"][subject]),
        susc_gradients["img_motion_free"][subject] < susc_threshold
    ) for subject in subjects
}

t2star_mae = {
    mask_key: {
        recon_key: {
            subject: [] for subject in reconstructions[mask_key][recon_key].keys()
        } for recon_key in reconstructions[mask_key].keys()
    } for mask_key in reconstructions.keys()
}

for mask_key in reconstructions.keys():
    for recon_key in recon_models.keys():
        for subject in config["subjects"]:
            tmp = calc_masked_MAE(
                t2star_maps[mask_key][recon_key][subject],
                t2star_maps_fully_sampled[subject],
                mask_susc[subject]
            )
            t2star_mae[mask_key][recon_key][subject] = tmp

save_path=(f"{config['out_folder']}/images_for_figures/validation/"
           f"KeepCenterQuant.png")
fig, axs = plt.subplots(3, 1, figsize=(4, 6), sharey=True, sharex=True)
sfs, lfs = adapt_font_size(6)
positions = np.arange(len(recon_models.keys()))
cols = ["#BEBEBE",  "#6C8B57"]
alphas = [0.7, 0.9]
for mask_key, ax in zip(reconstructions.keys(), axs):
    vp = ax.violinplot(
        [list(t2star_mae[mask_key][recon_key][subject])
         for recon_key in recon_models.keys()],
        positions=positions, widths=0.4,
        showmeans=True, showmedians=False, showextrema=False
    )
    for v, col, al in zip(vp['bodies'], cols, alphas):
        v.set_facecolor(col)
        v.set_edgecolor(col)
        v.set_alpha(al)
        v.set_zorder(2)
    vp['cmeans'].set_edgecolor("#E8E8E8")
    ax.set_ylabel("MAE [ms]", fontsize=sfs)
    ax.set_title(central_lines[mask_key], fontsize=sfs)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(positions,
                  [recon_labels[recon_key] for recon_key in recon_models.keys()],
                  fontsize=sfs)
    ax.tick_params(axis='y', labelsize=sfs)
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--', zorder=0)
    ax.yaxis.grid(True, which='minor', color='lightgray', linestyle='--', zorder=0)
fig.subplots_adjust(left=0.25, right=0.97, top=0.93, bottom=0.12, wspace=0.5,
                    hspace=0.4)
if save_path is not None:
    fig.savefig(save_path.replace(".png", ".svg"), dpi=300,
                bbox_inches='tight')
plt.show()



""" 5. Look at loss functions for different motion levels: """
Loss_ECC = ModelFitError(mask_image=True, error_type="emp_corr",
                         perform_bgf_corr=False)
Loss_SR = ModelFitError(mask_image=True, error_type="squared_residuals",
                        perform_bgf_corr=False)
Loss_AR = ModelFitError(mask_image=True, error_type="absolute_residuals",
                        perform_bgf_corr=False)

loss_ecc = {"motion_free": [], "motion_low": [],
            "motion_mid": [], "motion_high": []}
loss_sr = {"motion_free": [], "motion_low": [],
           "motion_mid": [], "motion_high": []}
loss_ar = {"motion_free": [], "motion_low": [],
           "motion_mid": [], "motion_high": []}
loss_ecc_inferior = {"motion_free": [], "motion_low": [],
                     "motion_mid": [], "motion_high": []}


slice_range_file = "./projects/moco_t2star/configs/mrm/real_motion/slice_ranges.yaml"
with open(slice_range_file, 'r') as file:
    range_slice_nrs = yaml.safe_load(file)["range_slice_nrs"]

with torch.no_grad():
    for idx in range(0, data_dict["img_motion_free"]["SQ-struct-04-p1-low"].shape
                         [0]):
        data = torch.tensor \
            ([data_dict["img_motion_free"]["SQ-struct-04-p1-low"][idx]], dtype=torch.complex64)
        mask = torch.tensor \
            ([data_dict["brain_mask"]["SQ-struct-04-p1-low"][idx][None]], dtype=torch.float32)
        loss_ecc["motion_free"].append(Loss_ECC(data, mask=mask).item())
        loss_sr["motion_free"].append(Loss_SR(data, mask=mask).item())
        loss_ar["motion_free"].append(Loss_AR(data, mask=mask).item())
        if idx in range(range_slice_nrs["SQ-struct-04-p1-low"][0],
                     range_slice_nrs["SQ-struct-04-p1-low"][1]+1):
            loss_ecc_inferior["motion_free"].append(
                Loss_ECC(data, mask=mask).item()
            )

    for descr in ["low", "mid", "high"]:
        for subject in ["SQ-struct-04-p1-" +descr, "SQ-struct-04-p2-" +descr,
                        "SQ-struct-04-p3-" +descr, "SQ-struct-04-p4-" +descr]:
            for idx in range(0, data_dict["img_motion"][subject].shape[0]):
                data = torch.tensor(
                    [data_dict["img_motion"][subject][idx]],
                    dtype=torch.complex64)
                mask = torch.tensor(
                    [data_dict["brain_mask"][subject][idx][None]],
                    dtype=torch.float32)
                loss_ecc["motion_" +descr].append \
                    (Loss_ECC(data, mask=mask).item())
                loss_sr["motion_" +descr].append \
                    (Loss_SR(data, mask=mask).item())
                loss_ar["motion_" +descr].append \
                    (Loss_AR(data, mask=mask).item())
                if idx in range(range_slice_nrs[subject][0],
                                range_slice_nrs[subject][1]+1):
                    loss_ecc_inferior["motion_" +descr].append(
                        Loss_ECC(data, mask=mask).item()
                    )


plot_violin_loss_functions(loss_ecc, descrs=["motion_free", "motion_low",
                                             "motion_mid", "motion_high"],
                           save_path=f"{config['out_folder']}/images_for_figures"
                                     f"/validation/loss_ecc.png",
                           loss_2=loss_ecc_inferior)

plot_violin_loss_functions(loss_sr, descrs=["motion_free", "motion_low",
                                            "motion_mid", "motion_high"],
                           save_path=f"{config['out_folder']}/images_for_figures"
                                     f"/validation/suppl_loss_sr.png")
plot_violin_loss_functions(loss_ar, descrs=["motion_free", "motion_low",
                                            "motion_mid", "motion_high"],
                           save_path=f"{config['out_folder']}/images_for_figures"
                                     f"/validation/suppl_loss_ar.png")

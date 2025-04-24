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
from projects.moco_t2star.utils import detach_torch, load_motion_data

set_seed(2109)

parser = argparse.ArgumentParser(description='Evaluate MoCo performance')
parser.add_argument('--config_path',
                    type=str,
                    default='./projects/moco_t2star/mrm/config_moco_motionfree.yaml',
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
subjects = sorted(subjects)


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

data_dict.pop("img_motion_free")

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

""" 4. Calculate susceptibility gradients and T2* maps: """
susc_gradients = {key: {} for key in ["img_motion"]}
for subject in subjects:
    for key in ["img_motion"]:
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
t2star_maps = {key: {} for key in (["img_motion", "img_hrqr", "img_orba",
                                    "img_sld"]
                                   + list(config["experiments"]))}
for subject in subjects:
    for key in ["img_motion", "img_hrqr", "img_orba", "img_sld"]:
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


for subject in subjects:
    for i_key in t2star_maps.keys():
        # mask the (unregistered) data with the brain mask:
        bm = np.where(data_dict["brain_mask"][subject] > 0.5, 1, 0)
        filled_bm = []
        for i in range(0, bm.shape[0]):
            filled_bm.append(binary_fill_holes(bm[i]))
        filled_bm = np.array(filled_bm)
        t2star_maps[i_key][subject] = t2star_maps[i_key][subject] * filled_bm



""" 5. Evaluate the mean exclusion rate: """
mean_mask = {key: [] for key in list(config["experiments"])+["img_orba", "img_sld"]}
ratio_excluded_lines = {key: [] for key in (list(config["experiments"])
                                            +["img_orba", "img_sld"])}

for exp_name in config["experiments"]:
    for subject in subjects:
        mean_mask[exp_name].append(
            np.mean(data_dict["mask_phimo"][exp_name][subject])
        )

for key in ['mask_orba', 'mask_sld_thr']:
    for subject in subjects:
        mean_mask[key.replace("mask", "img").replace("_thr", "")].append(
            np.mean(data_dict[key][subject])
        )

for exp_name in config["experiments"]:
    for subject in subjects:
        ratio_excluded_lines[exp_name].append(
            np.sum(data_dict["mask_phimo"][exp_name][subject]<0.5)
            /data_dict["mask_phimo"][exp_name][subject].size
            *100)
for key in ['mask_orba', 'mask_sld_thr']:
    for subject in subjects:
        ratio_excluded_lines[key.replace("mask", "img").replace("_thr", "")].append(
            np.sum(data_dict[key][subject]<0.5)
            /data_dict[key][subject].size
            *100)

for key in mean_mask.keys():
    print(key, "#####################")
    print(f"Mean of the mask: \n{np.mean(mean_mask[key]):.3f}+-"
          f"{np.std(mean_mask[key]):.3f}\n{mean_mask[key]}")
    print(f"Mean ratio of excluded lines: \n{np.mean(ratio_excluded_lines[key]):.3f}"
          f"+-{np.std(ratio_excluded_lines[key]):.3f}\n{ratio_excluded_lines[key]}")

# Latex code for table:
latex_code = r"""
\begin{table*}[h!]
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{} & \textbf{ORBA} & \textbf{SLD} & \textbf{PHIMO (Proposed)} \\ \midrule
"""
latex_code += "\\textbf{Average value of exclusion masks}          & "
latex_code += " & ".join([f"${np.mean(mean_mask[key]):.3f} \\pm {np.std(mean_mask[key]):.3f}$" for key in ['img_orba', 'img_sld', 'AllSlices-NoKeepCenter', 'Proposed']]) + r" \\" + "\n"
# latex_code += "\\textbf{Value range}         & "
# latex_code += " & ".join([f"[{np.min(mean_mask[key]):.3f}, {np.max(mean_mask[key]):.3f}]" for key in ['img_orba', 'img_sld', 'AllSlices-NoKeepCenter', 'Proposed']]) + r" \\ \midrule" + "\n"
latex_code += "\\textbf{Fraction of excluded lines ($<$0.5)} & "
latex_code += " & ".join([f"${np.mean(ratio_excluded_lines[key]):.2f} \\pm {np.std(ratio_excluded_lines[key]):.2f}$" for key in ['img_orba', 'img_sld', 'AllSlices-NoKeepCenter', 'Proposed']]) + r" \\" + "\n"
# latex_code += "\\textbf{Value range: } & "
# latex_code += " & ".join([f"[{np.min(ratio_excluded_lines[key]):.2f}, {np.max(ratio_excluded_lines[key]):.2f}]" for key in ['img_orba', 'img_sld', 'AllSlices-NoKeepCenter', 'Proposed']]) + r" \\" + "\n"
latex_code += r"""
\bottomrule
\end{tabular}
\caption{Comparison of predicted masks by ORBA, SLD and PHIMO for motion-free images.}
\label{tab:metrics_comparison}
\end{table*}
"""
print(latex_code)


""" 6. Look at example images: """
slice_ind = 16
for subject in ["SQ-struct-00-motionfree", "SQ-struct-46-motionfree"]:
    outfolder = f"{config['out_folder']}/images_for_figures/example_images_motionfree/"
    os.makedirs(outfolder, exist_ok=True)
    ind = np.where(data_dict["slices_ind"][subject] == slice_ind)[0][0]

    data_shape = np.array(t2star_maps["img_motion"][subject][ind].T.shape)
    cut_right = 10
    data_shape[1] -= cut_right
    fig, axs = plt.subplots(1, 6, figsize=(6*2, 2*data_shape[0]/data_shape[1]))
    for descr, save, ax in zip(["img_motion", "img_orba", "img_sld",
                                "AllSlices-NoKeepCenter", "Proposed", "img_hrqr"],
                               ["motion", "orba", "sld","AllSlices-NoKeepCenter",
                                "Proposed", "hrqr"],
                               axs):
        add_imshow_axis(ax, t2star_maps[descr][subject][ind].T[:, cut_right//2:-cut_right//2],
                        vmin=20, vmax=100, replace_nan=True)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.savefig(f"{outfolder}combined_t2star_{subject}_slice_{slice_ind}.png", dpi=400)
    plt.show()

    individual_imshow(data_dict["mask_phimo"]["Proposed"][subject][ind],
                      save_path=f"{outfolder}mask_phimo_Proposed_{subject}"
                                f"_slice_{slice_ind}.png", vmin=0, vmax=1)
    individual_imshow(data_dict["mask_phimo"]["AllSlices-NoKeepCenter"][subject][ind],
                      save_path=f"{outfolder}mask_phimo_AllSlices-NoKeepCenter_{subject}"
                                f"_slice_{slice_ind}.png", vmin=0, vmax=1)
    individual_imshow(data_dict["mask_orba"][subject][ind],
                      save_path=f"{outfolder}mask_orba_{subject}"
                                f"_slice_{slice_ind}.png", vmin=0, vmax=1)
    individual_imshow(data_dict["mask_sld_thr"][subject][ind],
                      save_path=f"{outfolder}mask_sld_{subject}"
                                f"_slice_{slice_ind}.png", vmin=0, vmax=1)


print("Done")

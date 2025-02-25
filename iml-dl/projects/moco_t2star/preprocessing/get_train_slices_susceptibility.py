import torch
import math
from dl_utils.config_utils import set_seed
from mr_utils.parameter_fitting import T2StarFit, CalcSusceptibilityGradient
from projects.moco_t2star.utils import *

set_seed(2109)

config = {
    'out_folder': './results/MRM/',
    'data_params': {'path': './data/links_to_data/recon_test_motion/',
                    'only_bm_slices': True,
                    'bm_thr': 0.2,
                    'normalize': 'percentile_image',
                    'normalize_volume': True,
                    'min_slice_nr': {'default': 0},
                    'simulated_data': False
                    }
}

subjects = glob.glob(f'{config["data_params"]["path"]}/raw_data/**fV4.mat')
subjects = sorted([os.path.basename(s).split("_nr")[0] for s in subjects])


""" 1. Load the motion-corrupted data: """
keys = ["sens_maps", "img_motion", 'img_motion_free', "brain_mask", "gray_matter",
        "white_matter", "mask_gt", "slices_ind", "filename_motion_free", "img_hrqr"]
data_dict = {key: {sub: [] for sub in subjects} for key in keys}

for subject in subjects:
    data_dict = load_motion_data(subject, config, data_dict)

for key in data_dict.keys():
    for subject in data_dict[key].keys():
        data_dict[key][subject] = np.array(data_dict[key][subject])


""" 1a Look at susceptibility gradients: """
susc_gradients = {key: {} for key in ["img_motion", "img_motion_free"]}
for subject in subjects:
    for key in ["img_motion", "img_motion_free"]:
        susc_gradients[key][subject] = []
        calc_susc_gradient = CalcSusceptibilityGradient(
            complex_img=torch.tensor(np.rollaxis(data_dict[key][subject], 1, 4), dtype=torch.complex64),
            mask=torch.tensor(data_dict["brain_mask"][subject], dtype=torch.float32),
            mode="inference"
        )
        susc_grad_x, susc_grad_y, susc_grad_z = calc_susc_gradient()
        susc_gradients[key][subject] = np.sqrt(
            detach_torch(susc_grad_x) ** 2
            + detach_torch(susc_grad_y) ** 2
            + detach_torch(susc_grad_z) ** 2
        )

# plot example susceptibility gradient maps:
slices = [0, 3, 6, 9]
fig, axes = plt.subplots(2, len(slices), figsize=(15, 6))
for i, slice_idx in enumerate(slices):
    im1 = axes[0, i].imshow(susc_gradients["img_motion"][subjects[0]][slice_idx].T,
                            cmap="gray", vmin=0, vmax=150)
    axes[0, i].set_title(f"Motion, Slice "
                         f"{data_dict['slices_ind'][subjects[0]][slice_idx]}")
    axes[0, i].axis("off")
    fig.colorbar(im1, ax=axes[0, i])
    im2 = axes[1, i].imshow(susc_gradients["img_motion_free"][subjects[0]][slice_idx].T,
                            cmap="gray", vmin=0, vmax=150)
    axes[1, i].set_title(f"Motion-Free, Slice "
                         f"{data_dict['slices_ind'][subjects[0]][slice_idx]}")
    axes[1, i].axis("off")
    fig.colorbar(im2, ax=axes[1, i])
plt.show()

# calculate the mean susceptibility gradient values for each slice across the brain mask:
susc_gradients_mean = {key: {} for key in ["img_motion", "img_motion_free"]}
for key in ["img_motion", "img_motion_free"]:
    for subject in subjects:
        masked_array = np.ma.masked_where(data_dict["brain_mask"][subject] == 0, susc_gradients[key][subject])
        susc_gradients_mean[key][subject] = np.mean(masked_array, axis=(1, 2))

# plot the mean susceptibility gradient values for each slice:
num_subjects = len(subjects)
num_cols = math.ceil(math.sqrt(num_subjects))
num_rows = math.ceil(num_subjects / num_cols)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()
for ax, subject in zip(axes, subjects):
    for key, color in zip(["img_motion", "img_motion_free"], ["tab:blue", "tab:gray"]):
        ax.plot(data_dict['slices_ind'][subject],
                susc_gradients_mean[key][subject], label=key, color=color)
    ax.set_xlabel("Slice Index")
    ax.set_ylabel("Mean Susceptibility Gradient")
    ax.legend()
    ax.set_title(f"Subject: {subject}")
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.5)
    ax.grid(axis='x', which='major', linestyle='--', linewidth=0.5)
    ax.axhspan(20, 80, color='tab:green', alpha=0.15)
for i in range(num_subjects, len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.show()

# Print all slice numbers, where the mean susceptibility
# gradient is smaller 80 for consecutive slices (max. 10 slices):
key = "img_motion"
max_nr_slices = 8
for subject in subjects:
    slices = data_dict['slices_ind'][subject]
    condition = susc_gradients_mean[key][subject] < 80
    longest_patch = []
    current_patch = []

    for i, slice_idx in enumerate(slices):
        if condition[i]:
            current_patch.append(slice_idx)
            if len(current_patch) > len(longest_patch):
                longest_patch = current_patch
        else:
            current_patch = []

    patch = (longest_patch[0:max_nr_slices] if len(longest_patch) > max_nr_slices
             else longest_patch)
    print("{}: [{}, {}]".format(subject, patch[0], patch[-1]))



print("Done")


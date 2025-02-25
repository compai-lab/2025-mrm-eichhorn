import os
import subprocess
import h5py
import torch
import merlinth
import numpy as np
import wandb
import matplotlib.pyplot as plt
import ants
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import piq
from lpips import LPIPS
import itertools
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from dl_utils.config_utils import import_module
from optim.losses.ln_losses import L2
from optim.losses.image_losses import SSIM_Magn, PSNR_Magn
from optim.losses.physics_losses import T2StarDiff
from data.t2star_loader import *


def create_dir(folder):
    """Create a directory if it does not exist."""

    if not os.path.exists(folder):
        os.makedirs(folder)
    return 0


def detach_torch(data):
    """Detach torch data and convert to numpy."""

    return (data.detach().cpu().numpy()
            if isinstance(data, torch.Tensor) else data)


def process_input_data(device, data):
    """Processes input data for training."""

    img_cc_fs = data[0].to(device)
    sens_maps = data[1].to(device)
    img_cc_fs_gt = data[2].to(device)
    img_hrqrmoco = data[3].to(device)
    mask_simulation = data[4].to(device)
    brain_mask = data[5].to(device)
    brain_mask_noCSF = data[6].to(device)
    filename = data[7]
    slice_num = data[8].unsqueeze(1).to(torch.float32).to(device)
    simulated_data = torch.unique(data[11])

    return (img_cc_fs, sens_maps, img_cc_fs_gt, img_hrqrmoco, brain_mask,
            brain_mask_noCSF, filename, slice_num, simulated_data,
            mask_simulation)


def round_differentiable(x):
    """Round a tensor in a differentiable way."""

    return x + x.round().detach() - x.detach()


def load_recon_model(recon_dict, device):
    """Load the pretrained reconstruction model."""

    model_class = import_module(recon_dict['module_name'],
                                recon_dict['class_name'])
    recon_model = model_class(**(recon_dict['params']))

    checkpoint = torch.load(recon_dict['weights'],
                            map_location=torch.device(device))
    recon_model.load_state_dict(checkpoint['model_weights'])

    return recon_model.to(device).eval()


def apply_undersampling_mask_torch(img, sens_maps, mask,
                                   keep_central_point=False,
                                   add_mean_fs=False):
    """Apply the given undersampling mask to the given image."""

    coil_imgs = img.unsqueeze(2) * sens_maps
    kspace = merlinth.layers.mri.fft2c(coil_imgs)

    # reshape the mask:
    mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1).repeat(
        1, kspace.shape[1], kspace.shape[2], 1, kspace.shape[-1])

    if keep_central_point:
        # if the mask excludes central k-space lines, adapt it to still keep
        # the central point
        npe, nfe = coil_imgs.shape[3], coil_imgs.shape[4]
        if npe % 2 == 0:
            if nfe % 2 == 0:
                mask[:, :, :, npe//2-1:npe//2+1, nfe//2-1:nfe//2+1] = 1
            else:
                mask[:, :, :, npe//2-1:npe//2+1, nfe//2] = 1
        else:
            if nfe % 2 == 0:
                mask[:, :, :, npe//2, nfe//2-1:nfe//2+1] = 1
            else:
                mask[:, :, :, npe//2, nfe//2] = 1

    coil_imgs_zf = merlinth.layers.mri.ifft2c(kspace * mask)

    if add_mean_fs:
        # if the mask excludes central k-space lines, add the mean of the fully
        # sampled image to the zero-filled image
        npe = coil_imgs.shape[3]
        if npe % 2 == 0:
            condition = mask[:, :, :, npe//2-1:npe//2+1].sum() == 0
        else:
            condition = mask[:, :, npe//2].sum() == 0
        if condition:
            coil_imgs_zf = (coil_imgs_zf
                            + torch.mean(coil_imgs, dim=(3,4), keepdim=True)
                            )

    img_cc_zf = torch.sum(coil_imgs_zf * torch.conj(sens_maps), dim=2)

    A = merlinth.layers.mri.MulticoilForwardOp(
        center=True,
        channel_dim_defined=False
    )
    kspace_zf = A(img_cc_zf, mask, sens_maps)

    return img_cc_zf, kspace_zf, mask.to(img_cc_zf.dtype)


def perform_reconstruction(img, sens_maps, mask_reduced, recon_model,
                           keep_central_point=False, add_mean_fs=False):
    """Perform reconstruction using the pretrained reconstruction model.

    Note: If the reconstruction model is a hypernetwork, setup needs to be
    changed and inference needs to be done for each slice individually.
    """

    img_zf, kspace_zf, mask = apply_undersampling_mask_torch(
        img, sens_maps, mask_reduced, keep_central_point=keep_central_point,
        add_mean_fs=add_mean_fs
    )

    return recon_model(img_zf, kspace_zf, mask, sens_maps), img_zf, mask


def prepare_for_logging(data):
    """Detach the data and crop the third dimension if necessary"""

    data_prepared = detach_torch(data)

    return (data_prepared[:, :, 56:-56] if data_prepared.shape[2] > 112
            else data_prepared)


def convert2wandb(data, abs_max_value, min_value, media_type="video",
                  caption=""):
    """
    Convert normalized data to a format suitable for logging in WandB.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    abs_max_value : np.ndarray
        Maximum absolute value for normalization.
    min_value : float
        Minimum value for normalization.
    media_type : str, optional
        Type of media ("video" or "image"). Default is "video".
    caption : str, optional
        Caption for the logged data. Default is "".

    Returns
    -------
    wandb.Video or wandb.Image
        Formatted data for WandB logging.
    """

    if media_type == "video":
        if np.amin(min_value) < 0:
            return wandb.Video(
                ((np.swapaxes(data[:, None], -2, -1)
                  / abs_max_value[:, None, None, None]+1) * 127.5
                 ).astype(np.uint8),
                fps=0.5, caption=caption
            )
        else:
            return wandb.Video(
                (np.swapaxes(data[:, None], -2, -1)
                 / abs_max_value[:, None, None, None] * 255
                 ).astype(np.uint8),
                fps=0.5, caption=caption
            )
    if media_type == "image":
        if np.amin(min_value) < 0:
            return wandb.Image(
                (np.swapaxes(data[0], -2, -1)
                 / abs_max_value + 1) * 127.5,
                caption=caption
            )
        else:
            return wandb.Image(
                np.swapaxes(data[0], -2, -1) / abs_max_value * 255,
                caption=caption
            )



def log_images_to_wandb(prediction_example, ground_truth_example,
                        motion_example, mask_example, second_mask_example=None,
                        third_mask_example=None, hr_qr_example=None,
                        wandb_log_name="Examples", slice=0,
                        captions=None, data_types=None):
    """Log data to WandB for visualization"""

    prediction_example = prepare_for_logging(prediction_example)
    ground_truth_example = prepare_for_logging(ground_truth_example)
    motion_example = prepare_for_logging(motion_example)
    if hr_qr_example is not None:
        hr_qr_example = prepare_for_logging(hr_qr_example)
    mask_example = detach_torch(mask_example)
    if second_mask_example is not None:
        second_mask_example = detach_torch(second_mask_example)
    if third_mask_example is not None:
        third_mask_example = detach_torch(third_mask_example)

    if data_types is None:
        data_types = ["magn", "phase"]
    if captions is None:
        captions = ["PHIMO", "Motion-free", "Motion-corrupted"]
    if hr_qr_example is not None:
        captions.append("HR/QR-MoCo")
    data_operations = {
        "magn": np.abs,
        "phase": np.angle,
        "real": np.real,
        "imag": np.imag
    }

    excl_rate = np.round(1 - np.sum(abs(mask_example)) / mask_example.size, 2)

    for data_type in data_types:
        pred, gt, motion = map(data_operations[data_type],
                           [prediction_example, ground_truth_example,
                            motion_example])
        if hr_qr_example is not None:
            hr_qr = data_operations[data_type](hr_qr_example)

        # Max / Min values for normalization:
        max_value = np.nanmax(np.abs(np.array([pred, gt, motion])),
                              axis=(0, 2, 3))
        min_value = np.nanmin(np.abs(np.array([pred, gt, motion])),
                              axis=(0, 2, 3))

        # Track multi/single-echo data as video/image data:
        if prediction_example.shape[0] > 1:
            pred = convert2wandb(pred, max_value, min_value,
                                 media_type="video",
                                 caption=captions[0])
            gt = convert2wandb(gt, max_value, min_value,
                               media_type="video",
                               caption=captions[1])
            motion = convert2wandb(motion, max_value, min_value,
                               media_type="video",
                               caption=captions[2])
            if len(mask_example.shape) == 1:
                mask = np.repeat(mask_example[None, None, :, None], 75, 3)
            elif len(mask_example.shape) == 4:
                mask = mask_example[0, 0][None, None, :, 18:93]
            mask = wandb.Video((mask*255).astype(np.uint8),
                               fps=0.5,
                               caption="Pred. Mask ({})".format(excl_rate))
            if second_mask_example is not None:
                mask2 = np.repeat(second_mask_example[None, None, :, None], 75, 3)
                mask2 = wandb.Video((mask2*255).astype(np.uint8),
                                    fps=0.5,
                                    caption="Mask from simulation")
            if third_mask_example is not None:
                mask3 = np.repeat(third_mask_example[None, None, :, None], 75, 3)
                mask3 = wandb.Video((mask3*255).astype(np.uint8),
                                    fps=0.5,
                                    caption="Finetuning mask")
            if hr_qr_example is not None:
                hr_qr = convert2wandb(hr_qr, max_value, min_value,
                                        media_type="video",
                                        caption=captions[3])
        else:
            pred = convert2wandb(pred, max_value, min_value,
                                 media_type="image",
                                 caption=captions[0])
            gt = convert2wandb(gt, max_value, min_value,
                               media_type="image", caption=captions[1])
            motion = convert2wandb(motion, max_value, min_value,
                               media_type="image", caption=captions[2])
            mask = np.repeat(mask_example[:, None], 112, 1)
            mask = wandb.Image(np.swapaxes((mask*255).astype(np.uint8),
                                           -2, -1),
                               caption="Pred. Mask ({})".format(excl_rate))
            if hr_qr_example is not None:
                hr_qr = convert2wandb(hr_qr, max_value, min_value,
                                        media_type="image",
                                        caption=captions[3])

        log_key = "{}{}/slice_{}".format(wandb_log_name, data_type, slice)
        log_data = [motion, pred, gt]
        if hr_qr_example is not None:
            log_data.append(hr_qr)
        log_data.append(mask)
        if second_mask_example is not None:
            log_data.append(mask2)
        if third_mask_example is not None:
            log_data.append(mask3)
        wandb.log({log_key: log_data})


def determine_echoes_to_exclude(mask):
    """Determine the number of echoes to exclude based on the mask."""

    exclusion_rate = 1 - torch.sum(mask) / mask.numel()

    leave_last_dict = {0.08: 2, 0.18: 3, 0.28: 4}
    leave_last = None

    for key in sorted(leave_last_dict.keys()):
        if exclusion_rate >= key:
            leave_last = leave_last_dict[key]

    return leave_last


def log_t2star_maps_to_wandb(t2star_pred, t2star_gt,
                             t2star_motion, t2star_hrqr=None,
                             wandb_log_name="Examples", slice=0):
    """Log data to WandB for visualization"""

    figure_size = (5, 2.5)
    t2stars = [t2star_motion, t2star_pred, t2star_gt]
    titles = ["Motion-corrupted", "PHIMO", "Motion-free"]

    if t2star_hrqr is not None:
        figure_size = (6.5, 2.5)
        t2stars.append(t2star_hrqr)
        titles.append("HR/QR-MoCo")

    fig = plt.figure(figsize=figure_size, dpi=300)
    min_value = 0
    max_value = 200

    for nr, (map, title) in enumerate(zip(t2stars, titles)):
        plt.subplot(1, len(titles)+1, nr + 1)
        plt.imshow(map.T, vmin=min_value, vmax=max_value, cmap='gray')
        plt.axis("off")
        plt.title(title, fontsize=8)
        if nr == len(titles)-1:
            cax = plt.axes([0.75, 0.3, 0.025, 0.35])
            cbar = plt.colorbar(cax=cax)
            cbar.ax.tick_params(labelsize=8)

    log_key = "{}slice_{}".format(wandb_log_name, slice)
    wandb.log({log_key: fig})


def calculate_img_metrics(target, data, bm, metrics_to_be_calc,
                          include_brainmask=True):
    """ Calculate metrics for a given target array and data array."""

    metrics = {}
    methods_dict = {
        'MSE': L2(mask_image=include_brainmask),
        'SSIM': SSIM_Magn(mask_image=include_brainmask, channel=target.shape[1]),
        'PSNR': PSNR_Magn(mask_image=include_brainmask)
    }

    for descr in metrics_to_be_calc:
        for m in methods_dict:
            if descr.startswith(m):
                metric = methods_dict[m]
                break
        if "magn" in descr:
            metrics[descr] = metric(
                torch.abs(target), torch.abs(data),
                bm
            ).item()
        elif "phase" in descr:
            metrics[descr] = metric(
                torch.angle(target), torch.angle(data),
                bm
            ).item()

    return metrics

def calculate_t2star_metrics(target, data, bm, metrics_to_be_calc):

    metrics = {}
    methods_dict = {
        'T2s_MAE': T2StarDiff
    }

    for descr in metrics_to_be_calc:
        for m in methods_dict:
            if descr.startswith(m):
                metric = methods_dict[m]()
                break
        metrics[descr] = metric(
            target, data, bm
        ).item()

    return metrics


def calculate_mask_metrics(target, data, metrics_to_be_calc):
    """Calculate metrics for a given target and predicted mask."""

    metrics = {}
    for descr in metrics_to_be_calc:
        if "MAE" in descr:
            pe_center_start = target.shape[1] // 4
            pe_center_end = 3 * target.shape[1] // 4

            central_mae = torch.mean(
                torch.abs(target[:, pe_center_start:pe_center_end]
                          - data[:, pe_center_start:pe_center_end])
            )
            peripheral_mae = torch.mean(
                torch.abs(torch.cat((
                    target[:, :pe_center_start]
                    - data[:, :pe_center_start],
                    target[:, pe_center_end:]
                    - data[:, pe_center_end:]),
                    dim=1
                ))
            )

            metrics["MAE_central"] = central_mae.item()
            metrics["MAE_peripheral"] = peripheral_mae.item()

    return metrics


def rigid_registration(fixed, moving, *images_to_move,
                       inv_reg=None, numpy=False, inplane=True):
    """Perform rigid registration of moving to fixed image."""

    if not numpy:
        device = fixed.device
        # convert to numpy:
        fixed = detach_torch(fixed)
        moving = detach_torch(moving)
        images_to_move = [detach_torch(im) if not isinstance(im, np.ndarray)
                          else im for im in images_to_move]

    # calculate transform for fixed and moving
    fixed_image = ants.from_numpy(abs(fixed))
    moving_image = ants.from_numpy(abs(moving))

    # Perform registration
    registration_result = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform="Rigid",
        random_seed=2019
    )

    # apply it to the other images
    images_reg = []
    if inplane:
        for im in images_to_move:
            im_reg = np.zeros_like(im)
            for i in range(len(im)):
                ants_image = ants.from_numpy(im[i])

                # Apply the transformation to the image
                im_reg[i] = ants.apply_transforms(
                    fixed_image,
                    moving=ants_image,
                    transformlist=registration_result['fwdtransforms'],
                    interpolator="bSpline"
                ).numpy()
            images_reg.append(im_reg)
    else:
        for im in images_to_move:
            im_reg = np.zeros_like(im)
            for i in range(im.shape[1]):
                ants_image = ants.from_numpy(im[:, i])

                # Apply the transformation to the image
                im_reg[:, i] = ants.apply_transforms(
                    fixed_image,
                    moving=ants_image,
                    transformlist=registration_result['fwdtransforms'],
                    interpolator="bSpline"
                ).numpy()
            images_reg.append(im_reg)

    if inv_reg is not None:
        if inplane:
            inv_reg_ = []
            for i in range(len(inv_reg)):
                ants_image = ants.from_numpy(inv_reg[i].astype(float))
                # Apply the transformation to the mask
                inv_reg_slice = ants.apply_transforms(
                    fixed_image,
                    moving=ants_image,
                    transformlist=registration_result['invtransforms'],
                    interpolator="bSpline"
                ).numpy()
                inv_reg_.append(inv_reg_slice)
        else:
            ants_image = ants.from_numpy(inv_reg.astype(float))
            # Apply the transformation to the mask
            inv_reg_ = ants.apply_transforms(
                fixed_image,
                moving=ants_image,
                transformlist=registration_result['invtransforms'],
                interpolator="bSpline"
            ).numpy()

        if not numpy:
            output = [torch.tensor(im).to(device) for im in images_reg].append(
                torch.tensor(inv_reg_).to(device))
            return output
        else:
            images_reg.append(inv_reg_)
            return images_reg
    else:
            if not numpy:
                return [torch.tensor(im).to(device) for im in images_reg]
            else:
                return images_reg


def reg_data_to_gt(img_gt, img, t2star,
                   inplane=True,  inv_reg=None):
    """Register image and T2star map to ground truth image."""

    if inplane:
        img_reg, t2star_reg, inv_reg_ = [], [], []
        for i in range(len(img_gt)):
            reg_result = rigid_registration(abs(img_gt[i])[0],
                                            abs(img[i])[0],
                                            abs(img)[i],
                                            [t2star[i]],
                                            inv_reg=inv_reg,
                                            numpy=True,
                                            inplane=inplane)
            img_reg.append(reg_result[0])
            t2star_reg.append(reg_result[1][0])
            if inv_reg is not None:
                inv_reg_ .append(reg_result[2])
    else:
       reg_result = rigid_registration(
           abs(img_gt)[:, 0],
           abs(img)[:, 0],
           abs(img),
           t2star[:, None],
           inv_reg=inv_reg,
           numpy=True,
           inplane=inplane
       )
       img_reg = reg_result[0]
       t2star_reg = reg_result[1][:, 0]
       if inv_reg is not None:
           inv_reg_ = reg_result[2]

    if inv_reg is None:
        return np.array(img_reg), np.array(t2star_reg)
    else:
        return np.array(img_reg), np.array(t2star_reg), np.array(inv_reg_)


def save_as_nii_with_header(image_array, header, output_path):
    """
    Save a numpy image array as a NIfTI file with a specified header.

    Parameters:
    - image_array: numpy array representing the image data.
    - header: Nifti1Header object with custom header information.
    - output_path: path to save the .nii file.
    """

    img = nib.Nifti1Image(np.rollaxis(image_array[:, ::-1, ::-1], 0, 3), affine=None, header=header)
    img.header["scl_slope"] = 1
    img.header["scl_inter"] = 0
    nib.save(img, output_path)


def run_matlab_script(script_name, *args):
    """
    Run a MATLAB script with specified arguments from Python using subprocess.

    Parameters:
    - script_name: Name of the MATLAB script (without .m extension).
    - *args: Additional arguments to pass to the script.
    """

    args_str = ', '.join(f"'{arg}'" for arg in args)
    matlab_cmd = f"matlab -batch \"addpath('{os.path.dirname(script_name)}'); {os.path.basename(script_name)}({args_str})\""
    print(matlab_cmd)

    try:
        # Run the MATLAB command
        result = subprocess.run(matlab_cmd, shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)

        # Print standard output and error (optional)
        print("MATLAB Output:", result.stdout)
        print("MATLAB Error:", result.stderr if result.stderr else "No errors.")

    except subprocess.CalledProcessError as e:
        print("An error occurred:", e.stderr)


def reg_data_to_ref_spm(img_ref, img, other_images, img_inv=None, header=None,
                        tmp_dir=None):

    # save the images as niftis with the header of the reference nii
    os.makedirs(tmp_dir, exist_ok=True)
    save_as_nii_with_header(img_ref, header,
                            os.path.join(tmp_dir, 'img_ref.nii'))
    save_as_nii_with_header(img, header,
                            os.path.join(tmp_dir, 'img.nii'))
    for nr, other in enumerate(other_images):
        save_as_nii_with_header(other, header,
                                os.path.join(tmp_dir, f'img_other_{nr}.nii'))
    if img_inv is not None:
        save_as_nii_with_header(img_inv, header,
                                os.path.join(tmp_dir, 'img_inv.nii'))

    # call matlab script to perform the registration
    run_matlab_script("./projects/moco_t2star/registration_matlab/spm_coregister",
                      os.path.join(tmp_dir, 'img_ref.nii'),
                      os.path.join(tmp_dir, 'img.nii'),
                      *([os.path.join(tmp_dir, f'img_other_{nr}.nii')
                         for nr in range(len(other_images))])
                      )

    if img_inv is not None:
        # if an inverse transform is to be performed, call the registration
        # script with flipped input
        run_matlab_script("./projects/moco_t2star/registration_matlab/spm_coregister",
                          os.path.join(tmp_dir, 'img.nii'),
                          os.path.join(tmp_dir, 'img_ref.nii'),
                          os.path.join(tmp_dir, 'img_inv.nii')
                          )

    # load the registered images
    other_images_reg = []
    for nr, other in enumerate(other_images):
        tmp = nib.load(os.path.join(tmp_dir, f'rimg_other_{nr}.nii')).get_fdata()
        other_images_reg.append(np.rollaxis(tmp, 2, 0)[:, ::-1, ::-1])

    if img_inv is not None:
        inv_reg = nib.load(os.path.join(tmp_dir, 'rimg_inv.nii')).get_fdata()
        inv_reg = np.rollaxis(inv_reg, 2, 0)[:, ::-1, ::-1]

    # delete the whole temporary directory
    os.system(f"rm -r {tmp_dir}")

    if len(other_images_reg) == 1:
        other_images_reg = other_images_reg[0]

    if img_inv is not None:
        return other_images_reg, inv_reg
    else:
        return other_images_reg


def save_dict_to_hdf5(group, d):
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, create a subgroup and recurse
            subgroup = group.create_group(key)
            save_dict_to_hdf5(subgroup, value)
        elif value is None:
            continue
        else:
            # Otherwise, store the value as a dataset
            group.create_dataset(key, data=value)


def load_dict_from_hdf5(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            # If it's a subgroup, recursively load its items
            result[key] = load_dict_from_hdf5(item)
        else:
            # Otherwise, it's a dataset, so we store the value
            result[key] = item[()]
    return result


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        folder = folder+"_1"
        check_folder(folder)
    return folder


def calc_masked_MAE(img1, img2, mask):
    """Calculate Mean Absolute Error between two images in a specified mask"""

    masked_diff = np.ma.masked_array(abs(img1 - img2),
                                     mask=(mask[:, None] != 1))
    return np.mean(masked_diff, axis=(1, 2)).filled(0)


def calc_masked_SSIM_3D(img, img_ref,  mask):
    """Calculate SSIM between two 3D images in a specified mask"""

    ssims = []
    for i in range(len(img)):
        _, ssim_values = structural_similarity(
            img_ref[i], img[i], data_range=np.amax(img_ref[i]),
            gaussian_weights=False, full=True
        )
        masked = np.ma.masked_array(ssim_values, mask=(mask[i] != 1))
        ssims.append(np.mean(masked))

    return np.array(ssims)


def calc_masked_FSIM_3D(img, img_ref, mask):
    """Calculate FSIM between two 3D images in a specified mask

    Note: The mask is multiplied to the images and not the FSIM values.
    The images are normalized to the maximum value of the reference image.
    """

    img = torch.from_numpy((img * mask)[:, None] / np.amax(img_ref)).float()
    img_ref = torch.from_numpy((img_ref * mask / np.amax(img_ref))[:, None]).float()
    fsims = piq.fsim(img, img_ref, data_range=1.0, reduction='none',
                     chromatic=False)
    return detach_torch(fsims)


def calc_masked_LPIPS_3D(img, img_ref, mask):
    """Calculate LPIPS between two 3D images in a specified mask.

    Note: The mask is multiplied to the images and not the FSIM values.
    The images are normalized to the maximum value of the reference image.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_vgg = LPIPS(net='vgg', verbose=False).to(device)

    img = torch.from_numpy((img * mask)[:, None] / np.amax(img_ref)).float() * 2 -1
    img_ref = torch.from_numpy((img_ref * mask / np.amax(img_ref))[:, None]).float() * 2 - 1
    img = img.repeat(1, 3, 1, 1).to(device)
    img_ref = img_ref.repeat(1, 3, 1, 1).to(device)
    loss_vgg = loss_fn_vgg(img, img_ref)

    return detach_torch(loss_vgg[:, 0, 0, 0])


def calc_masked_SSIM_4D(img, img_ref, mask, av_echoes=True,
                        later_echoes=False, normalize=True):
    """Calculate SSIM between two 4D images in a specified mask"""

    ssims = []
    for i in range(img.shape[0]):
        if normalize:
            ref = (img_ref[i] - np.mean(img_ref[i])) / np.std(img_ref[i])
            data = (img[i] - np.mean(img[i])) / np.std(img[i])
        else:
            ref = img_ref[i]
            data = img[i]

        ssim_echoes = []
        for j in range(data.shape[0]):
            mssim, ssim_values = structural_similarity(
                ref[j], data[j], data_range=np.amax(ref[j]),
                gaussian_weights=True, full=True
            )
            masked = np.ma.masked_array(ssim_values, mask=(mask[i] != 1))
            ssim_echoes.append(np.mean(masked))
        if av_echoes:
            ssims.append(np.mean(ssim_echoes))
        elif later_echoes:
            ssims.append(np.mean(ssim_echoes[-later_echoes:]))
        else:
            ssims.append(ssim_echoes)
    return np.array(ssims)


def calc_masked_PSNR_4D(img, img_ref, mask, av_echoes=True,
                        later_echoes=False, normalize=True):
    """Calculate PSNR between two 4D images in a specified mask"""

    psnrs = []
    for i in range(img.shape[0]):
        psnr_echoes = []
        if normalize:
            ref = (img_ref[i] - np.mean(img_ref[i])) / np.std(img_ref[i])
            data = (img[i] - np.mean(img[i])) / np.std(img[i])
        else:
            ref = img_ref[i]
            data = img[i]

        for j in range(data.shape[0]):
            d_ref = ref[j].flatten()[mask[i].flatten() > 0]
            d_img = data[j].flatten()[mask[i].flatten() > 0]
            psnr_echoes.append(
                peak_signal_noise_ratio(d_ref, d_img,
                                        data_range=np.amax(ref[j]))
            )
        if av_echoes:
            psnrs.append(np.mean(psnr_echoes))
        elif later_echoes:
            psnrs.append(np.mean(psnr_echoes[-later_echoes:]))
        else:
            psnrs.append(psnr_echoes)
    return np.array(psnrs)


def statistical_testing(keys, metric):
    """
    Perform statistical testing using Wilcoxon signed rank tests and multiple
    comparison correction (FDR) on metric values for pairs of keys.

    Parameters
    ----------
    keys : list
        Sorted keys of the metrics dictionary.
    metric : dict
        Dictionary containing metric values for each key.

    Returns
    -------
    tuple
        A tuple containing combinations and p-values for statistical testing.
    """

    combinations = list(itertools.combinations(np.arange(0, len(keys)),
                                               2))
    p_values = []
    for comb in combinations:
        p_values.append(wilcoxon(metric[keys[comb[0]]],
                                 metric[keys[comb[1]]],
                                 alternative='two-sided')[1])
    rej, p_values_cor, _, __ = multipletests(p_values, alpha=0.05,
                                             method='fdr_bh', is_sorted=False,
                                             returnsorted=False)

    insign = np.where(p_values_cor >= 0.05)
    for ins in insign[0]:
        print(keys[combinations[ins][0]],
              keys[combinations[ins][1]],
              " No significant difference.")

    return combinations, p_values_cor


def load_motion_data(subject, config, data_dict, load_aal3=False):
    print("Loading data for subject", subject)
    Dataset = RawMotionT2starDataset(
        select_one_scan=subject,
        load_whole_set=False,
        **config['data_params']
    )

    filename_move, filename_gt, _ = Dataset.raw_samples[0]
    slices_ind = sorted(Dataset.get_slice_indices(filename_move))
    data_dict["filename_motion_free"][subject] = filename_gt

    normalize = config['data_params']['normalize']
    normalize_volume = config['data_params']['normalize_volume']

    (sens_maps, img_cc_fs,
     img_cc_fs_gt, img_hr_qr, mask_simulation) = load_h5_data_motion(
        filename_move, filename_gt,
        simulated_data=config["data_params"]["simulated_data"],
        normalize=normalize,
        normalize_volume=normalize_volume,
    )
    segmentations = load_segmentations_nii(
            filename_gt, load_aal3=load_aal3
        )
    if load_aal3:
        brain_mask, gray_matter, white_matter, subregions = segmentations
    else:
        brain_mask, gray_matter, white_matter = segmentations

    for idx in slices_ind:
        cc_fs_gt = img_cc_fs_gt[idx]
        cc_fs = img_cc_fs[idx]
        hr_qr = img_hr_qr[:, idx]

        if not normalize_volume:
            cc_fs_gt = normalize_images(cc_fs_gt, [cc_fs_gt],
                                        normalization_method=normalize)[0]
            cc_fs = normalize_images(cc_fs, [cc_fs],
                                     normalization_method=normalize)[0]
            hr_qr = normalize_images(hr_qr, [hr_qr],
                                            normalization_method=normalize)[0]

        data_dict["sens_maps"][subject].append(equalize_coil_dimensions(
            sens_maps[idx]
        ))
        data_dict["img_motion"][subject].append(cc_fs)
        data_dict["img_motion_free"][subject].append(cc_fs_gt)
        data_dict["img_hrqr"][subject].append(hr_qr)
        data_dict["brain_mask"][subject].append(brain_mask[:, :, idx])
        data_dict["gray_matter"][subject].append(gray_matter[:, :, idx])
        data_dict["white_matter"][subject].append(white_matter[:, :, idx])
        data_dict["slices_ind"][subject].append(idx)
        if load_aal3:
            data_dict["subregions"][subject].append(subregions[:, :, idx])

        if mask_simulation is not None:
            data_dict['mask_gt'][subject].append(mask_simulation[idx])
        else:
            for tmp in ["SQ-struct-00", "SQ-struct-44", "SQ-struct-47",
                        "SQ-struct-48", "SQ-struct-01", "SQ-struct-02",
                        "SQ-struct-03", "SQ-struct-04",
                        "SQ-struct-04-p1-low", "SQ-struct-04-p1-mid",
                        "SQ-struct-04-p1-high", "SQ-struct-04-p2-low",
                        "SQ-struct-04-p2-mid", "SQ-struct-04-p2-high",
                        "SQ-struct-04-p3-low", "SQ-struct-04-p3-mid",
                        "SQ-struct-04-p3-high", "SQ-struct-04-p4-low",
                        "SQ-struct-04-p4-mid", "SQ-struct-04-p4-high"]:
                if tmp + "_" in filename_move:
                    logging.info("Loading the mask from the motion timing "
                                 "experiment for comparison.")
                    tmp = np.loadtxt(
                        f"/home/iml/hannah.eichhorn/Data/mqBOLD/"
                        f"RawYoungHealthyVol/motion_timing/{subject}/mask.txt",
                        unpack=True).T
                    # shift to match the correct timing:
                    tmp = np.roll(tmp, 3, axis=1)
                    tmp[:, 0:3] = 1
                    mask_timing = np.take(tmp, idx, axis=0)
                    data_dict['mask_gt'][subject].append(mask_timing)

    return data_dict

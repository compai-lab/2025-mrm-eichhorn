"""
Save the simulated test data in the BIDS format for the mqBOLD processing.
"""

import glob
import os
import shutil
import h5py
import numpy as np
import nibabel as nib

from data.t2star_loader import compute_coil_combined_reconstructions


def save_as_bids(data, folder_bids, folder_scanner_bids, acq="fullres",
                 sim_nr_save=None):

    # bring data into shape (12, 112, 112, 36):
    data = np.rollaxis(data, 0, 4)[:, ::-1, ::-1]
    pad_width = ((0, 0), (10, 10), (0, 0), (0, 0))
    data = np.pad(data, pad_width, mode='constant')

    # load the still nifti reconstructed on the scanner to extract the affine
    # as well as max amplitude
    filename_bids = glob.glob(f"{folder_bids}**_echo-01_task-still_acq-{acq}_"
                              f"T2star.nii.gz")[0]
    scanner_nii_affine = nib.load(filename_bids).affine

    # load the still nifti to extract the max amplitude for rescaling
    data_bids = np.zeros((12, 36, 112, 112))
    for i in range(0, 12):
        file_real = filename_bids.replace("_task", "_real_task")
        file_imag = filename_bids.replace("_task", "_imaginary_task")
        offset = 2047
        data_bids[i] = abs((np.rollaxis(
            nib.load(file_real).dataobj.get_unscaled(), 2, 0) - offset) + \
                           1j * (np.rollaxis(
            nib.load(file_imag).dataobj.get_unscaled(), 2, 0) - offset))
    max_bids = np.amax(data_bids)
    data = data / np.amax(abs(data)) * max_bids

    if sim_nr_save is not None:
        filename_bids = filename_bids.replace(sim_nr_save[0], sim_nr_save[1])
        folder_bids = folder_bids.replace(sim_nr_save[0], sim_nr_save[1])
        os.makedirs(folder_bids, exist_ok=True)

    # go through echoes and save the data:
    for i in range(data.shape[0]):
        if i < 9:
            echo_nr = "0" + str(i + 1)
        else:
            echo_nr = str(i + 1)
        # real, imaginary and magnitude separately:
        offset = 2047
        nii_abs = nib.Nifti1Image(abs(data[i]), scanner_nii_affine)
        filename_save = filename_bids.replace(
            "echo-01", f"echo-{echo_nr}"
        ).replace("task-still", "task-move")
        nib.save(nii_abs, filename_save)

        # subtract offset for real and imaginary data:
        nii_real = nib.Nifti1Image(np.real(data[i]) + offset,
                                   scanner_nii_affine)
        nib.save(nii_real,
                 filename_save.replace("_task", "_real_task"))
        nii_imag = nib.Nifti1Image(np.imag(data[i]) + offset,
                                   scanner_nii_affine)
        nib.save(nii_imag,
                 filename_save.replace("_task", "_imaginary_task"))

    # copy the json files from scanner recon:
    file_subst = os.path.basename(filename_bids).replace(
        "echo-01", f"echo-**"
    ).replace("task-still", "task-move").replace(
        ".nii.gz", ".json"
    )
    # remove -sim0* from the filename:
    sim_nr = ""
    for i in range(0, 10):
        if f"-sim0{i}" in file_subst:
            sim_nr = f"-sim0{i}"
            file_subst = file_subst.replace(f"-sim0{i}", "")

    json_files = glob.glob(f"{folder_scanner_bids}/"
                           f"{file_subst}")
    for json_file in json_files:
        shutil.copy(json_file,
                    json_file.replace(
                        folder_scanner_bids, folder_bids
                    ).replace("_echo", f"{sim_nr}_echo"))


in_folder = ("/PATH/TO/SIMULATED/DATA/") # insert folder with simulated data
bids_from_raw_folder = ("/PATH/TO/BIDS/FROM/RAW/DATA/input/") # insert folder
# with bids data that were converted from raw data (as reference)

subjects_base = ["SQ-struct-38", "SQ-struct-45", "SQ-struct-33",
                 "SQ-struct-43", "SQ-struct-46", "SQ-struct-00",
                 "SQ-struct-44", "SQ-struct-47","SQ-struct-48"]
subjects = []
for sub in subjects_base:
    subjects.append(sub+"-sim00")
    subjects.append(sub+"-sim01")
    subjects.append(sub+"-sim02")

for subject in subjects:
    # copy anatomy data from motion-free data:
    subject_base = os.path.basename(subject).split("-sim")[0]
    anatomy_folder = f"{bids_from_raw_folder}/sub-{subject_base}/anat/"
    anatomy_files = glob.glob(f"{anatomy_folder}/*")
    os.makedirs(anatomy_folder.replace(
        "bids_from_raw", "bids_from_raw_simulated"
    ).replace(subject_base, subject), exist_ok=True)
    for file in anatomy_files:
        shutil.copy(file, file.replace(
            "bids_from_raw", "bids_from_raw_simulated"
        ).replace(subject_base, subject))

    # copy still data from motion-free data:
    still_folder = f"{bids_from_raw_folder}/sub-{subject_base}/t2star/"
    still_folder_new = still_folder.replace(
        "bids_from_raw", "bids_from_raw_simulated"
    ).replace(subject_base, subject)
    os.makedirs(still_folder_new, exist_ok=True)
    for still_file in glob.glob(f"{still_folder}/*task-still*"):
        still_file_new = still_file.replace(
            "bids_from_raw", "bids_from_raw_simulated"
        ).replace(subject_base, subject)
        shutil.copy(still_file, still_file_new)

    # save motion-corrupted data:
    filename_move = glob.glob(f"{in_folder}/*{subject}/*move*.h5")[0]
    with h5py.File(filename_move, "r") as f:
        kspace = f["RawData"]["data"][:]
        sens_maps = f["RawData"]["sens_maps"][:]
        motion_data = {}
        for key in f["MotionSimulation"]["MotionData"].keys():
            motion_data[key] = f["MotionSimulation"]["MotionData"][key][:]
        scan_order_path = f["MotionSimulation"]["ScanOrder"][()].decode("utf-8")
        img_data_shape = f["RawData"]["data"][:, :, 0].shape

    img_cc_fs = compute_coil_combined_reconstructions(
        kspace, sens_maps, y_shift=0, remove_oversampling=False
    )
    save_as_bids(img_cc_fs,  folder_bids=still_folder_new,
                 folder_scanner_bids=still_folder, acq="fullres")

    # save HR and QR data for other simulation subjects:
    rotate_sim_nrs_hr = {"-sim00": "-sim02",
                      "-sim01": "-sim00",
                      "-sim02": "-sim01"}
    rotate_sim_nrs_qr = {"-sim00": "-sim01",
                        "-sim01": "-sim02",
                        "-sim02": "-sim00"}
    sim_key = ""
    for key in rotate_sim_nrs_hr.keys():
        if key in still_folder_new:
            sim_key = key
    if sim_key == "":
        raise ValueError("No key found")

    kspace_hr = np.zeros_like(kspace)
    kspace_qr = np.zeros_like(kspace)
    kspace_hr[:, :, :, 92//4:92//4+92//2] = kspace[:, :, :, 92//4:92//4+92//2]
    kspace_qr[:, :, :, 94//8*3:94//8*3+94//4] = kspace[:, :, :, 94//8*3:94//8*3+94//4]

    img_cc_fs_hr = compute_coil_combined_reconstructions(
        kspace_hr, sens_maps, y_shift=0, remove_oversampling=False
    )
    save_as_bids(img_cc_fs_hr, folder_bids=still_folder_new,
                 folder_scanner_bids=still_folder, acq="MoCoHR",
                 sim_nr_save=[sim_key, rotate_sim_nrs_hr[sim_key]])
    img_cc_fs_qr = compute_coil_combined_reconstructions(
        kspace_qr, sens_maps, y_shift=0, remove_oversampling=False
    )
    save_as_bids(img_cc_fs_qr, folder_bids=still_folder_new,
                 folder_scanner_bids=still_folder, acq="MoCoQR",
                 sim_nr_save= [sim_key, rotate_sim_nrs_qr[sim_key]])

print("Done")
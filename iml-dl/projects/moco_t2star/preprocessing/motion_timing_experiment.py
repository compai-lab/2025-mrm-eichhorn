"""
Script for transforming the motion timing instructions for the motion
timing experiment into a corresponding exclusion mask.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py as h5


def create_scan_order_file(h5_file, out_file):
    """
    Generate text file containing the acquisition order.

    Parameters
    ----------
    h5_file : str
        Path to the raw data file in h5 format (exported with MRecon).
    out_file : str
        Filename under which the scan order file should be saved.
    """

    # Load h5 file
    with h5.File(h5_file, 'r') as f:
        label_look_up = f['out']['Labels']['LabelLookupTable'][:]
    if len(label_look_up.shape) == 8:
        label_look_up = label_look_up[:, :, 0, 0, 0, 0, :, 0]
    if len(label_look_up.shape) == 7:
        label_look_up = label_look_up[:, 0, 0, 0, 0, :, 0][None]

    flattened_labels = label_look_up.flatten()
    sorted_indices = np.argsort(flattened_labels)
    rank_array = np.empty_like(sorted_indices)
    rank_array[sorted_indices] = np.arange(len(flattened_labels))
    acq_order = rank_array.reshape(label_look_up.shape)

    nr_slices, nr_echoes, nr_pes = acq_order.shape
    total_acquisitions = nr_slices * nr_echoes *  nr_pes
    contrasts, slices, ys = [], [], []

    # Iterate through each possible time point (acquisition rank)
    for time_point in range(total_acquisitions):
        # Find the index of the current time point in the flattened array
        sl, e, y = np.where(acq_order == time_point)
        # Append the coordinates to their respective lists
        contrasts.append(e.item())
        slices.append(sl.item())
        ys.append(y.item())

    # Convert lists to arrays if necessary
    contrasts = np.array(contrasts)
    slices = np.array(slices)
    ys = np.array(ys)

    for i in range(0, 100):
        print(i, contrasts[i], slices[i], ys[i])

    reps = np.zeros_like(contrasts)

    # add acquisition time defined by TR and echo times:
    rep_time = 2.3
    echo_time = 0.005001
    echo_diff = 0.005
    all_echoes = echo_time + (nr_echoes - 1) * echo_diff
    gap = (rep_time - all_echoes * nr_slices) / nr_slices

    times = []
    curr_time = 0
    for i in range(0, nr_pes):
        for j in range(0, nr_slices):
            for k in range(0, nr_echoes):
                if k == 0:
                    curr_time += echo_time
                else:
                    curr_time += echo_diff
                times.append(curr_time)
            curr_time += gap
        curr_time = rep_time * (i + 1)

    # exclude all data with slice > nr_slices:
    reps = reps[slices < nr_slices]
    contrasts = contrasts[slices < nr_slices]
    ys = ys[slices < nr_slices]
    slices = slices[slices < nr_slices]

    save_arr = np.array([times, reps, contrasts, slices, ys]).T
    np.savetxt(out_file.replace(".txt", f"{nr_slices}_slices.txt"), save_arr,
               header='Timing, repetition, contrast, slice, phase encode step y')


scan_order_file = ("./projects/motion_simulation/configs/scan_orders/"
                   "36_slices.txt")

time, _, echo, sl, pe = np.loadtxt(scan_order_file, unpack=True)

# insert manually:
subject = "SQ-struct-04-p4-low"
start_timing = [5, 50, 100, 165, 195]
end_timing = [10, 55, 110, 170, 205]

mask = np.ones_like(time)
for s, e in zip(start_timing, end_timing):
    mask[(time >= s) & (time < e)] = 0

mask = np.round(mask.reshape(-1, 12).mean(axis=1))
sl = sl[::12]
pe = pe[::12]

# Create a structured array with 'sl' and 'pe' as fields
structured_array = np.array(list(zip(sl, pe, mask)),
                            dtype=[('sl', 'f8'), ('pe', 'f8'), ('mask', 'f8')])

# Sort by 'sl' first, then by 'pe'
sorted_array = np.sort(structured_array, order=['sl', 'pe'])

# Extract the sorted 'sl', 'pe', and 'mask' arrays
sl_sorted = sorted_array['sl']
pe_sorted = sorted_array['pe']
mask_sorted = sorted_array['mask']

final_mask = mask_sorted.reshape(-1, 92)
np.savetxt("/PATH/TO/DATA/motion_timing/{}/mask.txt".format(subject),
           final_mask, fmt='%d')
# insert the original data path here


plt.figure(figsize=(112/20, 92/20))
plt.imshow(np.repeat(final_mask[0][None], 112, axis=0).T, cmap='gray')
plt.xticks([], [])
plt.show()

print("Done!")
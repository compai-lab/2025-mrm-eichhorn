import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from mr_utils.motion_transforms import (transf_from_parameters,
                                        parameters_from_transf,
                                        transform_sphere)
from scipy.interpolate import interp1d
import scipy.signal as signal
from sklearn.decomposition import PCA


def load_motion_data(files, curve_types):
    """
    Load motion data from files and store in curve_types dictionary.

    Parameters
    ----------
    files : list
        Filenames to load motion data from.
    curve_types : dict
        Dictionary containing curve types as keys and empty lists as values.

    Returns
    -------
    curve_types : dict
        Updated dictionary containing curve types as keys and lists of motion
        data as values.
    """

    for filename in files:
        motion_data = np.loadtxt(filename)
        for curve_suffix, curve_dict in curve_types.items():
            if curve_suffix in filename:
                T, R = motion_data[:, :3], motion_data[:, 3:] * 180 / np.pi
                for t, i in zip(['t_x', 't_y', 't_z'], [0, 1, 2]):
                    curve_dict[t].append(T[:, i])
                for r, i in zip(['r_x', 'r_y', 'r_z'], [0, 1, 2]):
                    curve_dict[r].append(R[:, i])
                curve_dict['filename'].append(filename)
    return curve_types


def convert_to_arrays(curve_types):
    """Convert lists of motion data to numpy arrays."""

    empty_keys = []
    for key, curve_type in curve_types.items():
        for t, curves in curve_type.items():
            if len(curves) != 0:
                if t != 'filename':
                    max_length = max(len(lst) for lst in curves)
                    # Pad shorter lists with np.nan
                    curves = [curve.tolist() + [np.nan]*(max_length - len(curve))
                              for curve in curves]
                    curve_type[t] = np.array(curves)
            else:
                empty_keys.append(key)
                break
    for key in empty_keys:
        curve_types.pop(key)
    return curve_types


def print_curve_counts(curve_types):
    """Print the number of curves for each curve type."""

    print('#####################')
    for curve_suffix, curve_dict in curve_types.items():
        print(f'Number of {curve_suffix}: {len(curve_dict["t_x"])}')
    print('#####################')


def interpolate_curves(key, curve_types, repetition_times, length_all,
                       all_curves=None):
    """
    Interpolate motion curve to have uniform length / timings.

    Parameters
    ----------
    key : str
        Curve type key.
    curve_types : dict
        Dictionary containing curve types as keys and lists of motion data as
        values.
    repetition_times : dict
        Dictionary containing repetition times for each curve type.
    length_all : int
        Length of the interpolated curves.
    all_curves : dict, optional
        Dictionary containing all interpolated curves.

    Returns
    -------
    all_curves : dict
        Updated dictionary containing all interpolated curves.
    """

    if all_curves is None:
        all_curves = {'t_x': [], 't_y': [], 't_z': [],
                       'r_x': [], 'r_y': [], 'r_z': []}
    curves = curve_types[key]
    for t, curve_list in curves.items():
        if t != 'filename':
            tmp = curve_list
            if key != '-fMRI':
                seconds = np.arange(0, len(tmp[0]) * repetition_times[key],
                                    repetition_times[key])
                seconds_all = np.arange(0, len(tmp[0]) * repetition_times[key],
                                        repetition_times['-fMRI'])
                tmp_itp = interp1d(seconds, tmp)
                tmp_itp = tmp_itp(seconds_all)
            else:
                tmp_itp = tmp
            length = len(tmp_itp[0])
            for i in range(0, int(length / length_all)):
                for a in tmp_itp:
                    curve_segment = a[i * length_all:length_all + i * length_all]
                    if not np.any(np.isnan(curve_segment)):
                        all_curves[t].append(curve_segment)
    return all_curves


def get_equal_length_curves(curve_types, length_all, repetition_times):
    """
    Combine and interpolate motion curves to have equal length.

    Parameters
    ----------
    curve_types : dict
        Dictionary containing curve types as keys and lists of motion data as
        values.
    length_all : int
        Length of the interpolated curves.
    repetition_times : dict
        Dictionary containing repetition times for each curve type.

    Returns
    -------
    all_curves : dict
        Dictionary containing all interpolated curves.
    """

    if "_dsc" in curve_types:
        for t in curve_types['_dsc']:
            if t not in ['filename']:
                tmp = curve_types['_dsc'][t]
                tmp_l = []
                for a, b in zip(tmp[::2], tmp[1::2]):
                    tmp_l.append(np.concatenate((a, b)))
                curve_types['_dsc'][t] = tmp_l

    curve_keys = curve_types.keys()

    all_curves = None
    for key in curve_keys:
        all_curves = interpolate_curves(key, curve_types, repetition_times,
                                        length_all, all_curves)
        print(f"Length of all curves (+ {key}): ", len(all_curves['t_x']))

    return all_curves


def correct_curves(all_curves):
    """
    Correct motion curves to start at the origin.

    Parameters
    ----------
    all_curves : dict
        Dictionary containing all interpolated curves.

    Returns
    -------
    all_curves : dict
        Updated dictionary containing corrected curves.
    """

    for i in range(0, len(all_curves['t_x'])):
        if all_curves['t_x'][i][0] != 0:
            T = np.concatenate(([all_curves['t_x'][i]],
                                [all_curves['t_y'][i]],
                                [all_curves['t_z'][i]]), axis=0).T
            R = np.concatenate(([all_curves['r_x'][i]],
                                [all_curves['r_y'][i]],
                                [all_curves['r_z'][i]]), axis=0).T
            matrices = np.zeros((len(T), 4, 4))
            for j in range(len(T)):
                matrices[j] = transf_from_parameters(T[j], R[j])
            tr_matrices = np.matmul(np.linalg.inv(matrices[0]), matrices)
            T_0, R_0 = np.zeros((len(T), 3)), np.zeros((len(T), 3))
            for j in range(len(T)):
                T_0[j], R_0[j] = parameters_from_transf(tr_matrices[j])
            all_curves['t_x'][i] = T_0[:, 0]
            all_curves['t_y'][i] = T_0[:, 1]
            all_curves['t_z'][i] = T_0[:, 2]
            all_curves['r_x'][i] = R_0[:, 0]
            all_curves['r_y'][i] = R_0[:, 1]
            all_curves['r_z'][i] = R_0[:, 2]
    return all_curves


def calculate_motion_statistics(all_curves_, dset_shape, pixel_spacing, radius,
                                motion_free_threshold=2.0):
    """
    Calculate motion statistics for the given motion curves.

    Parameters
    ----------
    all_curves : dict
        Dictionary containing all interpolated curves.
    dset_shape : tuple
        Shape of the dataset.
    pixel_spacing : tuple
        Pixel spacing of the dataset.
    radius : float
        Radius of the sphere for calculating magnitude displacement.
    motion_free_threshold : float, optional
        Threshold for motion free fraction.

    Returns
    -------
    RMS_displacement : np.array
        RMS displacement for each curve.
    motion_free : np.array
        Motion free fraction for each curve.
    max_displacement : np.array
        Maximum displacement for each curve.
    """

    RMS_displacement = []
    motion_free = []
    max_displacement = []

    all_curves = all_curves_.copy()

    if len(all_curves['t_x'].shape) == 1:
        for key in all_curves.keys():
            all_curves[key] = [all_curves[key]]

    for i in range(len(all_curves['t_x'])):
        motion_data = np.array([all_curves['t_z'][i],
                                all_curves['t_y'][i],
                                all_curves['t_x'][i],
                                all_curves['r_z'][i],
                                all_curves['r_y'][i],
                                all_curves['r_x'][i]]).T
        centroids, tr_coords = transform_sphere(dset_shape=dset_shape,
                                                motion_parameters=motion_data,
                                                pixel_spacing=pixel_spacing,
                                                radius=radius)
        ind_median_centroid = np.argmin(
            np.sqrt(np.sum((centroids - np.median(centroids, axis=0)) ** 2,
                           axis=1))
        )
        displacement = tr_coords - tr_coords[ind_median_centroid]
        magnitude_displacement = np.sqrt(displacement[:, :, 0] ** 2
                                         + displacement[:, :, 1] ** 2
                                         + displacement[:, :, 2] ** 2)
        RMS_displacement.append(np.mean(magnitude_displacement))
        magnitude_sphere = np.mean(magnitude_displacement, axis=1)
        motion_free.append(np.count_nonzero(magnitude_sphere < motion_free_threshold)
                           /len(magnitude_sphere))
        max_displacement.append(np.amax(magnitude_sphere))

    return (np.array(RMS_displacement), np.array(motion_free),
            np.array(max_displacement))


def save_curves(all_curves, rms_displacement, motion_free, max_displacement,
                output_dir, dataset=None, repetition_times=None):
    """
    Save motion curves to JSON files.

    Parameters
    ----------
    all_curves : dict
        Dictionary containing all interpolated curves.
    rms_displacement : np.array
        RMS displacement for each curve.
    motion_free : np.array
        Motion free fraction for each curve.
    max_displacement : np.array
        Maximum displacement for each curve.
    output_dir : str
        Output directory.
    dataset : str
        Dataset name.
    repetition_times : dict
        Dictionary containing repetition times for each curve type.
    """

    if repetition_times is not None:
        time = np.arange(0, len(all_curves['t_x'][0]) * repetition_times["-fMRI"],
                         repetition_times["-fMRI"]).tolist()
    else:
        if "time" not in all_curves:
            raise ValueError("No time information available.")

    for i in range(len(all_curves['t_x'])):
        curve_dict = {
            "RMS_displacement": rms_displacement[i].tolist(),
            "motion_free": motion_free[i].tolist(),
            "max_displacement": max_displacement[i].tolist(),
            "t_x": all_curves['t_x'][i].tolist(),
            "t_y": all_curves['t_y'][i].tolist(),
            "t_z": all_curves['t_z'][i].tolist(),
            "r_x": all_curves['r_x'][i].tolist(),
            "r_y": all_curves['r_y'][i].tolist(),
            "r_z": all_curves['r_z'][i].tolist()
        }
        if repetition_times is not None:
            curve_dict["time"] = time
        else:
            curve_dict["time"] = all_curves["time"][i]

        if "filename" in all_curves and all_curves["filename"][i]:
            filename = all_curves["filename"][i]
        else:
            filename = f"{dataset}_{i:03}.json"

        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(curve_dict, f)


def save_single_curve(curve, filename):
    """
    Save a single motion curve to a JSON file.

    Parameters
    ----------
    curve : dict
        Motion curve to be saved.
    filename : str
        Filename of the output file.
    """

    # convert numpy arrays to lists
    for key, value in curve.items():
        if isinstance(value, np.ndarray):
            curve[key] = value.tolist()

    with open(filename, 'w') as f:
        json.dump(curve, f)


def load_curves(directory, items="metrics", filenames=None):
    """Load motion curves from JSON files."""

    if items == "metrics":
        items = ['RMS_displacement', 'motion_free', 'max_displacement']
    elif items == "all":
        items = ['RMS_displacement', 'motion_free', 'max_displacement',
                 't_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z', 'time']
    elif items == "motion_data":
        items = ['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z', 'time']
    else:
        raise ValueError("Invalid items argument.")

    curves = []

    if filenames is None:
        filenames = [f for f in os.listdir(directory) if f.endswith('.json')]

    for filename in filenames:
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                curve = json.load(f)
                tmp_dict = {'filename': filename}
                for item in items:
                    tmp_dict[item] = curve[item]
                curves.append(tmp_dict)
    return curves


def rank_curves(curves):
    """
    Rank curves based on RMS displacement, motion free, and max displacement.

    Parameters
    ----------
    curves : list
        Motion curves to be ranked.

    Returns
    -------
    curves : list
        Sorted list of curves.
    """

    # Rank curves for each metric
    rms_displacement_ranks = rankdata([curve['RMS_displacement'] for curve in curves])
    motion_free_ranks = rankdata([-curve['motion_free'] for curve in curves])
    max_displacement_ranks = rankdata([curve['max_displacement'] for curve in curves])

    # Calculate median rank for each curve
    for curve, rms_rank, motion_rank, max_rank in zip(curves, rms_displacement_ranks, motion_free_ranks, max_displacement_ranks):
        curve['median_rank'] = np.mean([rms_rank, motion_rank, max_rank])

    # Sort curves by median rank
    curves.sort(key=lambda x: x['median_rank'])

    return curves


def plot_curves(curves, title='Curves'):
    """
    Plot the RMS displacement, motion free, and max displacement values of the
    curves.

    Parameters
    ----------
    curves : list
        Motion curves to be plotted.
    title : str, optional
        Title of the plot.
    """

    # Extract the rms_displacement, motion_free, and max_displacement values
    rms_displacement = [curve['RMS_displacement'] for curve in curves]
    motion_free = [curve['motion_free'] for curve in curves]
    max_displacement = [curve['max_displacement'] for curve in curves]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(rms_displacement, label='RMS Displacement')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(motion_free, label='Motion Free')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(max_displacement, label='Max Displacement')
    plt.legend()
    plt.suptitle(title)
    plt.show()


def calculate_mean_metrics(sets):
    """Calculate the mean RMS displacement, motion free, and max displacement
    values for each set."""

    mean_metrics = {}

    for set_name, curves in sets.items():
        rms_displacement_mean = np.mean([curve['RMS_displacement'] for curve in curves])
        motion_free_mean = np.mean([curve['motion_free'] for curve in curves])
        max_displacement_mean = np.mean([curve['max_displacement'] for curve in curves])

        mean_metrics[set_name] = {
            'rms_displacement': rms_displacement_mean,
            'motion_free': motion_free_mean,
            'max_displacement': max_displacement_mean
        }

    return mean_metrics


def distribute_curves_into_sets(curves):
    """Distribute curves into training, validation, and test sets."""

    plot_curves(curves, title="Unsorted curves")

    # sort out the curves where motion_free is equal to 1.0:
    print("Number of all curves: {}".format(len(curves)))
    curves_motion_free = [curve for curve in curves if curve['motion_free'] > 0.93]
    curves = [curve for curve in curves if curve['motion_free'] <= 0.93]
    print("Number of curves with motion_free > 0.93: {}".format(len(curves_motion_free)))
    print("Number of curves with motion_free <= 0.93: {}".format(len(curves)))

    # Sort the curves after mean rank of individual metrics
    curves = rank_curves(curves)
    plot_curves(curves, title="Sorted curves")

    # Initialize sets
    sets = {'train': [], 'val': [], 'test': [], 'remaining': []}

    # Iteratively sample from all curves
    for i in range(len(curves)):
        if i % 4 == 0 or i % 4 == 1:
            sets['train'].append(curves[i])
        elif i % 4 == 2:
            sets['val'].append(curves[i])
        elif i % 4 == 3:
            sets['test'].append(curves[i])

    for curve in curves_motion_free:
        sets['remaining'].append(curve)

    mean_metrics = calculate_mean_metrics(sets)
    print("Mean metrics for each set while sorting:")
    for set_name, metrics in mean_metrics.items():
        print(f"Set: {set_name}")
        print(f"Mean RMS Displacement: {metrics['rms_displacement']}")
        print(f"Mean Motion Free: {metrics['motion_free']}")
        print(f"Mean Max Displacement: {metrics['max_displacement']}")
        print()

    # Randomly remove curves from 'val' to 'test' and 'training'
    while len(sets['val']) > 15:
        curve_to_move = random.choice(sets['val'])
        sets['val'].remove(curve_to_move)
        sets['test'].append(curve_to_move)

    while len(sets['test']) < 39:
        curve_to_move = random.choice(sets['train'])
        sets['train'].remove(curve_to_move)
        sets['test'].append(curve_to_move)

    while len(sets['test']) > 39:
        curve_to_move = random.choice(sets['test'])
        sets['test'].remove(curve_to_move)
        sets['train'].append(curve_to_move)

    mean_metrics = calculate_mean_metrics(sets)
    print("Mean metrics for each set after re-sorting:")
    for set_name, metrics in mean_metrics.items():
        print(f"Set: {set_name}")
        print(f"Mean RMS Displacement: {metrics['rms_displacement']}")
        print(f"Mean Motion Free: {metrics['motion_free']}")
        print(f"Mean Max Displacement: {metrics['max_displacement']}")
        print()

    return sets


def low_pass_filter(time_series, cutoff_freq, fs):
    """Apply a low-pass filter to the given time series."""

    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(1, normal_cutoff, btype='low', analog=False)
    filtered_series = signal.filtfilt(b, a, time_series)
    return filtered_series


def calculate_mean_curve(curves):
    """Calculate the mean motion curve for the given set of curves."""

    mean_curve = {'t_x': [], 't_y': [], 't_z': [],
                  'r_x': [], 'r_y': [], 'r_z': []}

    for t in ['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']:
        mean_curve[t] = np.mean([curve[t] for curve in curves], axis=0)

    mean_curve['time'] = curves[0]['time']

    return mean_curve


def subtract_curves(curve, mean_curve):
    """Subtract the mean curve from the given curve."""

    subtracted_curve = {}
    for t in ['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']:
        subtracted_curve[t] = (np.array(curve[t]) - np.array(mean_curve[t])).tolist()

    subtracted_curve['time'] = curve['time']

    return subtracted_curve


def perform_pca(curves):

    concatenated_curves = np.array([
        np.concatenate([curve['t_x'], curve['t_y'], curve['t_z'],
                        curve['r_x'], curve['r_y'], curve['r_z']])
        for curve in curves
    ])

    pca = PCA()
    pca.fit(concatenated_curves.T)
    pca_curves_tmp = pca.transform(concatenated_curves.T).T

    expl_var_train = pca.explained_variance_
    expl_var_rat_train = pca.explained_variance_ratio_

    # normalize the prinicipal components
    pca_curves_tmp = pca_curves_tmp.reshape(40, 6, len(curves[0]['t_x']))
    magn = np.rollaxis(
        np.repeat([np.sqrt(np.sum(pca_curves_tmp ** 2, axis=2))],
                  len(curves[0]['t_x']), axis=0),
        0, 3)
    pca_curves_tmp = pca_curves_tmp / (magn + 1e-8)

    # transform pca_train_tmp into a dictionary
    pca_curves = [None] * 40
    for i in range(pca_curves_tmp.shape[0]):
        pca_curves[i] = {}
        for j, t in enumerate(['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']):
            pca_curves[i][t] = pca_curves_tmp[i][j]

    return pca_curves, expl_var_train, expl_var_rat_train


def generate_pca_curve(pca_components, mean_curve, explained_variance,
                       weight_range=3):
    """Generate a PCA curve from the given PCA components and mean curve."""

    # Generate random weights and scale by explained variance
    weight = np.random.uniform(-weight_range, weight_range,
                               explained_variance.shape)
    variation = weight * np.sqrt(explained_variance)

    variation_sum = {
        t: np.sum([np.array(component[t]) * variation[i] for i, component in
                   enumerate(pca_components)], axis=0)
        for t in ['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']
    }

    pca_curve = {t: mean_curve[t] + variation_sum[t] for t in
                 ['t_x', 't_y', 't_z', 'r_x', 'r_y', 'r_z']}
    pca_curve['time'] = mean_curve['time']

    return pca_curve

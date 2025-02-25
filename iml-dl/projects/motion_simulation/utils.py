import logging
import os.path
import glob
from medutils.mri import mriForwardOp, mriAdjointOp
import numpy as np
import json
import argparse
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import copy
from skimage.morphology import erosion, dilation
from mr_utils.motion_transforms import *
from data.t2star_loader import load_h5_data, load_mask_from_nii
from data.t2star_loader import normalize_images_max, normalize_images_percentile


def load_config():
    """Load configuration file."""

    parser = argparse.ArgumentParser(description="Motion simulator")
    parser.add_argument("--config_path",
                        type=str,
                        default="projects/motion_simulation/configs/config_debug.yaml",
                        metavar="C",
                        help="path to configuration yaml file")
    args = parser.parse_args()
    with open(args.config_path, "r") as stream_file:
        config_file = yaml.load(stream_file, Loader=yaml.FullLoader)
    logging.info("[simulator::load_config] Loaded config file: {}".format(
        args.config_path)
    )
    return config_file, args.config_path


def generate_output_file_path(config_file, subject, j, motion_free_file):
    """Generate output file path for simulated data."""

    return "{}/{}-sim{:02d}/{}".format(
        config_file["output"]["simulated_data_folder"], subject, j,
        os.path.basename(motion_free_file).replace(".mat", ".h5").replace("sg_fV4", "sg_f_moveV4")
    )


def check_output_file(output_file):
    """Check if output file already exists."""

    if os.path.exists(output_file):
        logging.info("File already exists: {}. Skipping to the next motion curve.".format(output_file))
        return False
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        return True


def load_motion_data(filename, shift_by_median_position=True, data_shape=None,
                     pixel_spacing=(3.3, 2, 2), radius=64):
    """
    Load motion data from json file.

    Parameters
    ----------
    filename : str
        Path to the json file containing the motion data.
    shift_by_median_position : bool, optional
        Shift motion data by median position, by default True.
    data_shape : tuple, optional
        Shape of the data, by default None. Required if shift_by_median_position
        is True.
    pixel_spacing : tuple, optional
        Pixel spacing of the data, by default (3.3, 2, 2).
    radius : int, optional
        Radius of the sphere, by default 64.

    Returns
    -------
    dict
        Dictionary containing the motion data.
    """

    with open(os.path.join(filename), 'r') as f:
        data = json.load(f)
    logging.info("[simulator::load_motion_data] RMS displacement: {:.3f} mm, "
                 "max. displacement: {:.3f} mm, motion-free fraction: {:.3f}."
                 "".format(data["RMS_displacement"], data["max_displacement"],
                           data["motion_free"]))
    data.pop("RMS_displacement")
    data.pop("max_displacement")
    data.pop("motion_free")

    if shift_by_median_position:
        if data_shape is None:
            raise ValueError("data_shape must be provided if "
                             "shift_by_median_position is True.")
        motion_array = np.array([data[key] for key in ['t_z', 't_y', 't_x',
                                                       'r_z', 'r_y', 'r_x']]).T
        centroids, _ = transform_sphere(data_shape, motion_array, pixel_spacing,
                                        radius)
        index_median_centroid = np.argmin(np.sqrt(np.sum(
            (centroids - np.median(centroids, axis=0)) ** 2,
            axis=1
        )))

        matrices = np.zeros((len(data["time"]), 4, 4))
        for i in range(len(data["time"])):
            T = np.array([data["t_x"][i], data["t_y"][i], data["t_z"][i]]).T
            R = np.array([data["r_x"][i], data["r_y"][i], data["r_z"][i]]).T
            matrices[i] = transf_from_parameters(T, R)

        transformed_matrices = np.matmul(
            np.linalg.inv(matrices[index_median_centroid]),
            matrices
        )

        for i in range(len(data["time"])):
            t, r = parameters_from_transf(transformed_matrices[i])
            keys = ["t_x", "t_y", "t_z", "r_x", "r_y", "r_z"]
            for j, key in enumerate(keys):
                data[key][i] = t[j] if j < 3 else r[j - 3]

    return data


def load_raw_data(filename, normalize="abs_image"):
    """Load raw data from h5 file."""

    logging.info("[utils::load_raw_data] Loading motion-free data ...")

    with h5py.File(filename, "r") as hf:
        shape = hf['out']['Data'][:, :, 0, 0, :, 0].shape
        nr_slices = shape[0]

    sens_maps, img_cc_fs = None, None
    (sens_maps_slice, img_cc_fs_slice) = load_h5_data(
        filename, normalize=normalize, normalize_volume=False
    )
    for dataslice in range(0, nr_slices):
        if sens_maps is None:
            sens_maps = np.zeros((nr_slices,) + sens_maps_slice.shape,
                                 dtype=np.complex64)
            img_cc_fs = np.zeros((nr_slices,) + img_cc_fs_slice.cshape,
                                 dtype=np.complex64)
        sens_maps[dataslice] = sens_maps_slice
        img_cc_fs[dataslice] = img_cc_fs_slice
        if normalize == "abs_image":
            img_cc_fs = normalize_images_max( img_cc_fs, [img_cc_fs])[0]
        if normalize == "percentile_image":
            img_cc_fs = normalize_images_percentile(img_cc_fs, [img_cc_fs])[0]

    return img_cc_fs, sens_maps


def save_simulated_data(simulated_data, sens_maps, motion_data, mask,
                        motion_threshold, output_file, path_scan_order):
    """
    Save simulated data to h5 file.

    Note:
    - This function copies the motion-free data to the output file and
    substitutes/adds relevant fields with the simulated data in the correct
    format.
    - The Parameter YRange is adapted to prevent further y-shifts when
    loading the data again.

    Parameters
    ----------
    simulated_data : np.ndarray
        Simulated data to be saved.
    sens_maps : np.ndarray
        Sensitivity maps for converting image to kspace.
    motion_data : dict
        Dictionary containing the motion data to be saved.
    mask : np.ndarray
        Mask (created by thresholding the motion data) to be saved.
    motion_threshold : float
        Motion threshold used for the simulation.
    output_file : str
        Output file path.
    motion_free_file : str
        Motion-free data file path.
    """

    with h5py.File(output_file, "a") as hf:
        hf.create_group("MotionSimulation")

        # save simulated data in 'out/Data' (correct format)
        hf["MotionSimulation"]["Mask"] = mask
        hf["MotionSimulation"]["Mask_Threshold"] = motion_threshold
        hf["MotionSimulation"].create_group("MotionData")
        for key in motion_data.keys():
            hf["MotionSimulation"]["MotionData"][key] = motion_data[key]
        hf["MotionSimulation"]["ScanOrder"] = path_scan_order

        simulated_kspace = mriForwardOp(img=simulated_data[:, :, None],
                                       smaps=sens_maps,
                                       mask=1)
        hf.create_group("RawData")
        hf["RawData"]["data"] = simulated_kspace
        hf["RawData"]["sens_maps"] = sens_maps

    return 0



def load_brainmask(subject, raw_data_folder):
    """Load corresponding brain mask from nii file."""

    bm_file = glob.glob(os.path.join(
        os.path.dirname(raw_data_folder[:-1]),
        "bids_from_raw/output/",
        f"sub-{subject}",
        "task-still/qBOLD/T1w_coreg/rcBrMsk_CSF.nii"
    ))[0]

    return np.transpose(load_mask_from_nii(bm_file), (2, 0, 1))[:, None]


class SimulateMotion:
    def __init__(self,
                 motion_tracking,
                 motion_free_data,
                 sens_maps,
                 brain_mask,
                 path_scan_order,
                 include_b0_inhomogeneities=True,
                 threshold_motion_transform=False,
                 motion_threshold=0.5,
                 pixel_spacing=(3.3, 2, 2),
                 radius=64,
                 te_0=0.005001,
                 te_diff=0.005,
                 b0_range=None):
        super().__init__()

        self.motion_times = motion_tracking["time"]
        self.motion_parameters = np.array([motion_tracking[key]
                                           for key in ['t_z', 't_y', 't_x',
                                                       'r_z', 'r_y', 'r_x']]).T
        self.motion_free_data = motion_free_data
        self.sens_maps = sens_maps
        self.brain_mask = brain_mask

        self.scan_order = type('', (), {})()
        (self.scan_order.acquisition_times, self.scan_order.repetitions,
         self.scan_order.echoes, self.scan_order.slices,
         self.scan_order.pe_lines) = np.loadtxt(path_scan_order, unpack=True)

        self.include_b0_inhomogeneities = include_b0_inhomogeneities
        self.threshold_motion_transform = threshold_motion_transform
        self.motion_threshold = motion_threshold
        self.pixel_spacing = pixel_spacing
        self.te_0 = te_0
        self.te_diff = te_diff
        self.tes = np.arange(te_0,
                             te_0 + te_diff * self.motion_free_data.shape[1],
                             te_diff)
        if b0_range is None:
            self.b0_range = (2.0, 5.0)
        else:
            self.b0_range = b0_range

        self.magnitude_displacement, _ = calculate_average_displacement(
            self.motion_parameters, self.pixel_spacing,
            self.motion_free_data.shape, radius=radius
        )

    def calculate_exclusion_mask(self):
        """Calculate exclusion mask based on motion data."""

        reduced_mask, mask = displacement_to_mask(
            self.magnitude_displacement, self.motion_threshold,
            self.motion_free_data.shape, self.scan_order, self.motion_times
        )

        return self.magnitude_displacement, reduced_mask, mask


    @staticmethod
    def _apply_morphological_operations(mask):
        """Apply morphological operations to the mask."""

        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        mask_morph = np.zeros_like(mask)
        for i in range(len(mask)):
            tmp = dilation(mask[i], cross)
            tmp = dilation(tmp, cross)
            tmp = dilation(tmp, cross)
            tmp = erosion(tmp, cross)
            tmp = erosion(tmp, cross)
            tmp = erosion(tmp, cross)
            mask_morph[i] = tmp
        return mask_morph

    def _generate_simulated_field_map(self, times, data_shape, mask):
        """Generate random magnetic field inhomogeneities."""

        sim_field_map = np.zeros((len(times), data_shape[0], *data_shape[2:]))
        time_intervals = [((t - 1), t) for t in
                          range(1, int(times[-1]) + 1)]

        for t_min, t_max in time_intervals:
            intens_grad = self._simulate_varying_magnetic_field(data_shape, mask)
            for ind, ti in enumerate(times):
                if t_min <= ti < t_max:
                    sim_field_map[ind] = intens_grad

        return sim_field_map

    def _simulate_varying_magnetic_field(self, data_shape, mask, order=1):
        """Simulate random magnetic field for a given order."""

        half_shape = np.array(data_shape[2:]) / 2
        ranges = [np.arange(-n, n) + 0.5 for n in half_shape]
        x_mesh, y_mesh = np.asarray(np.meshgrid(*ranges, indexing='ij'))
        x_mesh = x_mesh.reshape(*x_mesh.shape, 1)
        y_mesh = y_mesh.reshape(*y_mesh.shape, 1)

        coefficients = np.random.random((order + 1, order + 1))
        x_orders, y_orders = np.meshgrid(np.arange(order + 1),
                                         np.arange(order + 1))

        # Filter out invalid combinations where x_order + y_order > order,
        # to reduce the complexity of the polynomial
        valid_combinations = x_orders + y_orders <= order

        intens_grad = np.sum(
            coefficients[valid_combinations]
            * x_mesh ** x_orders[valid_combinations]
            * y_mesh ** y_orders[valid_combinations],
            axis=2)

        for axis in [0, 1]:
            if np.random.randint(2):
                intens_grad = np.flip(intens_grad, axis=axis)

        intens_grad = np.tile(intens_grad, (data_shape[0], 1, 1))

        # Ensure that within brainmask the maximum intensity is
        # between 2 and 5 [Hz]
        max_int = np.amax(abs(intens_grad * mask[:, 0]))
        intens_grad /= max_int
        intens_grad *= np.random.uniform(self.b0_range[0], self.b0_range[1])

        return intens_grad

    def _create_random_B0_inhom(self, data_shape, motion_times):
        """Create random B0 inhomogeneities for each motion state."""

        mask = self._apply_morphological_operations(self.brain_mask[:, 0])
        mask = np.transpose(np.tile(mask, (12, 1, 1, 1)), (1, 0, 2, 3))

        sim_field_map = self._generate_simulated_field_map(motion_times,
                                                           data_shape,
                                                           mask)
        return sim_field_map

    def _simulate_one_motion_state(self, parameters, random_inhomogeneity):
        """
        Simulate motion for a single readout line.

        Parameters
        ----------
        parameters : np.ndarray
            Array with 6 motion parameters for the current readout line
            (3 translational and 3 rotational).
        random_inhomogeneity : np.ndarray
            Array with random motion-induced inhomogeneity field map.

        Returns
        -------
        np.ndarray
            Simulated k-space data for the current readout line.

        Notes
        -----
        In theory, the multiplication with B0 inhomogeneities is to be
        performed after rotating and translating the image. Here, we perform it
        before, since we multiply with randomly generated B0 inhomogeneities
        anyway. Like this, the brain mask does not need to be rigidly
        transformed together with the image.
        """

        if isinstance(random_inhomogeneity, np.ndarray):
            image = (copy.deepcopy(self.motion_free_data) * np.exp(
                -2j * np.pi * random_inhomogeneity[:, None]
                * self.tes[None, :, None, None]
            ).astype(np.complex64))
        elif (isinstance(random_inhomogeneity, bool)
              and random_inhomogeneity == False):
            image = copy.deepcopy(self.motion_free_data)
        else:
            raise ValueError(
                "random_inhomogeneity must be either 'False' or a numpy array.")

        for e in range(0, image.shape[1]):
            image[:, e] = apply_transform_image(image[:, e], parameters,
                                                self.pixel_spacing)

        kspace = mriForwardOp(img=image[:, :, None], smaps=self.sens_maps,
                             mask=1)

        return kspace.astype(np.complex64)

    def simulate_all_lines(self):
        """Simulate motion for all readout lines."""

        combined_kspace = mriForwardOp(img=self.motion_free_data[:, :, None],
                                       smaps=self.sens_maps,
                                       mask=1)

        if self.include_b0_inhomogeneities:
            inhomogeneities = self._create_random_B0_inhom(
                self.motion_free_data.shape, self.motion_times
            )
        else:
            inhomogeneities = None

        indices_above_threshold = np.where(
            self.magnitude_displacement > self.motion_threshold
        )[0]
        difference_motion_and_acquisition_times = np.abs(
            np.array(self.motion_times)[:, None] - self.scan_order.acquisition_times
        )
        motion_states = np.argmin(difference_motion_and_acquisition_times,
                                  axis=0)
        if self.threshold_motion_transform:
            indices_dict = {key: np.where(motion_states == key)[0] for key in
                            indices_above_threshold}
        else:
            indices_dict = {key: np.where(motion_states == key)[0] for key in
                            range(len(self.motion_times))}
        total_motion_states = len(indices_dict.keys())

        for count, motion_idx in enumerate(indices_dict.keys()):
            if self.include_b0_inhomogeneities and motion_idx in indices_above_threshold:
                random_inhomogeneity = inhomogeneities[motion_idx]
            else:
                random_inhomogeneity = False

            simulated_kspace = self._simulate_one_motion_state(
                parameters=self.motion_parameters[motion_idx],
                random_inhomogeneity=random_inhomogeneity
            )

            acq_indices = indices_dict[motion_idx]
            s = self.scan_order.slices[acq_indices].astype(int)
            e = self.scan_order.echoes[acq_indices].astype(int)
            pe = self.scan_order.pe_lines[acq_indices].astype(int)

            combined_kspace[s, e, :, pe] = simulated_kspace[s, e, :, pe]

            logging.info(
                "[SimulateMotion::simulate_all_lines] Simulated motion state"
                " # {} out of {} ...".format(count+1, total_motion_states)
            )

        combined_img = mriAdjointOp(kspace=combined_kspace,
                                    smaps=self.sens_maps,
                                    mask=np.ones_like(combined_kspace))
        return combined_img


def plot_simulation(simulated_data, motion_free_data,
                    save_plots=False, output_dir=None,
                    echoes_to_plot=None, slice=15):
    """Plot simulated data and motion-free data."""

    if echoes_to_plot is None:
        echoes_to_plot = [0, 5, 10]

    fig, axs = plt.subplots(len(echoes_to_plot), 2,
                            figsize=(10, 5*len(echoes_to_plot)))
    for i, e in enumerate(echoes_to_plot):
        axs[i, 0].imshow(np.abs(simulated_data[slice, e].T), cmap='gray')
        axs[i, 0].set_title(f'Simulated Data - Echo {e}')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(np.abs(motion_free_data[slice, e, :].T), cmap='gray')
        axs[i, 1].set_title(f'Motion Free Data - Echo {e}')
        axs[i, 1].axis('off')

    if save_plots:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'simulation.png'))
        else:
            logging.warning(
                "[simulator::plot_simulation] No output directory specified "
                "for saving plots."
            )

    plt.tight_layout()
    plt.show()


def plot_motion_parameters(motion_data, magnitude_displacements,
                           displacement_threshold,
                           save_plots=False, output_dir=None, ):
    """Plot motion parameters and highlight time points above threshold."""

    above_threshold = magnitude_displacements > displacement_threshold
    starts = np.where(
        np.diff(np.hstack(([False], above_threshold, [False])))
    )[0].reshape(-1, 2)
    gray_patch = mpatches.Patch(
        color='gray', alpha=0.5
    )

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for t in ['t_x', 't_y', 't_z']:
        axs[0].plot(motion_data['time'], motion_data[t], label=t)
    for start, end in starts:
        axs[0].axvspan(motion_data['time'][start],
                       motion_data['time'][end - 1],
                       color='gray', alpha=0.5)
    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(gray_patch)
    labels.append('d > {}'.format(displacement_threshold))
    axs[0].set_ylabel('Translation [mm]')
    axs[0].legend(handles, labels, loc="best", ncol=2)
    axs[0].set_xticks([])
    axs[0].axhline(0, color="gray")

    for t in ['r_x', 'r_y', 'r_z']:
        axs[1].plot(motion_data['time'], motion_data[t], label=t)
    for start, end in starts:
        axs[1].axvspan(motion_data['time'][start],
                       motion_data['time'][end - 1],
                       color='gray', alpha=0.5)
    handles, labels = axs[1].get_legend_handles_labels()
    handles.append(gray_patch)
    labels.append('d > {}'.format(displacement_threshold))
    axs[1].legend(handles, labels, loc="best", ncol=2)
    axs[1].set_ylabel('Rotation [deg]')
    axs[1].set_xlabel('Time [s]')
    axs[1].axhline(0, color="gray")

    plt.suptitle("Mean displacement: {:.3f} mm".format(
        np.mean(magnitude_displacements),
        displacement_threshold
    ))
    plt.tight_layout()
    if save_plots:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, 'motion_data.png'))
        else:
            logging.warning(
                "[simulator::plot_simulation] No output directory specified "
                "for saving plots."
            )
    plt.show()

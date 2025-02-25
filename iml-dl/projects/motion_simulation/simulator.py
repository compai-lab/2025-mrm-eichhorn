"""
simulator.py
- main entry point to start the motion simulation
- run this script (from iml-dl/.) with:

    python -u ./projects/motion-simulation/simulator.py
        --config_path path_to_config_file.yaml

Note: A config file needs to be generated first. An example of a config file
can be found under motion-simulation/configs/config_debug.yaml.
"""
import json
import shutil

import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
import logging
import warnings
import time
import os
import glob
from utils import *


def main():
    warnings.filterwarnings(action='ignore')
    logging.basicConfig(level=logging.INFO)

    config_file, config_path = load_config()
    logging.info("[simulator::main] Starting motion simulation. Results will "
                 "be saved in: "
                 "{}".format(config_file["output"]["simulated_data_folder"]))
    if (not config_file["simulation"]["include_motion_transform"]
            or not config_file["simulation"]["include_inhomogeneities"]):
        raise NotImplementedError(
            "Only include_motion_transform and include_inhomogeneities are "
            "implemented so far. Please set them to True in the config file."
        )

    subject_curve_mapping_file = ("./projects/motion_simulation/configs/"
                                  "data_set_split/{}{}").format(
        config_file["data_import"]["dataset_split"],
        config_file["data_import"]["subject_curve_mapping_file"]
    )
    with open(subject_curve_mapping_file, "r") as f:
        subject_curve_mapping = json.load(f)

    if "subjects" in config_file["data_import"]:
        subjects = config_file["data_import"]["subjects"]
    else:
        subjects = subject_curve_mapping.keys()
    total_subjects = len(subjects)
    current_date = time.strftime("%y-%m-%d_%H-%M")

    for i, subject in enumerate(subjects):
        logging.info("####################################")
        logging.info(
            "[simulator::main] Processing subject # {} out of {}: {}".format(
                i+1, total_subjects, subject))

        motion_free_file = glob.glob(
            "{}/{}/**sg_fV4**.mat".format(
                config_file["data_import"]["raw_data_folder"],
                subject
            ))[0]
        motion_free_data, sens_maps = load_raw_data(motion_free_file, normalize="None")
        norm_motion_free_data = np.max(np.abs(motion_free_data))
        motion_free_data /= norm_motion_free_data

        path_scan_order = ("./projects/motion_simulation/configs/"
                           f"scan_orders/{motion_free_data.shape[0]}_"
                           f"slices.txt")
        brain_mask = load_brainmask(subject,
                                    config_file["data_import"]["raw_data_folder"])

        nr_motion_curves = len(subject_curve_mapping[subject])
        for j, curve in enumerate(subject_curve_mapping[subject]):
            logging.info(
                "[simulator::main] Processing curve # {} out of {}: {}".format(
                    j+1, nr_motion_curves, curve))
            start = time.time()


            output_file = generate_output_file_path(config_file, subject, j,
                                                    motion_free_file)
            if not check_output_file(output_file):
                continue

            output_dir_plots = config_file["output"].get("output_dir_plots",
                                                         False)
            if output_dir_plots:
                output_dir_plots = os.path.join(
                    output_dir_plots,
                    current_date,
                    "{}-sim{:02d}".format(subject, j)
                )
                os.makedirs(output_dir_plots, exist_ok=True)
                dest_config_path = os.path.join(
                    os.path.dirname(os.path.normpath(output_dir_plots)),
                    os.path.basename(config_path)
                )
                if not os.path.exists(dest_config_path):
                    shutil.copy(config_path, dest_config_path)

            motion_data = load_motion_data(
                config_file["data_import"]["motion_data_folder"] + curve,
                config_file["data_import"].get("shift_by_median_position", True),
                data_shape=motion_free_data.shape,
                pixel_spacing=(3.3, 2, 2)
            )

            Simulation = SimulateMotion(
                motion_data,
                motion_free_data,
                sens_maps,
                brain_mask,
                path_scan_order=path_scan_order,
                threshold_motion_transform=config_file["simulation"]["threshold_motion_transform"],
                motion_threshold=config_file["simulation"]["motion_threshold"],
                pixel_spacing=(3.3, 2, 2),
                b0_range=config_file["simulation"].get("b0_range", None)
            )

            (magnitude_displacement, reduced_mask,
             mask) = Simulation.calculate_exclusion_mask()
            plot_motion_parameters(
                motion_data,
                magnitude_displacement,
                config_file["simulation"]["motion_threshold"],
                save_plots=config_file["output"].get("save_plots", False),
                output_dir=output_dir_plots
            )

            simulated_data = Simulation.simulate_all_lines()
            plot_simulation(simulated_data, motion_free_data,
                            save_plots=config_file["output"].get("save_plots", False),
                            output_dir=output_dir_plots,
                            echoes_to_plot=[0, 5, 10])

            save_simulated_data(simulated_data, sens_maps, motion_data, mask,
                                config_file["simulation"]["motion_threshold"],
                                output_file, path_scan_order)

            symlink_path = os.path.join(os.path.dirname(output_file),
                                        os.path.basename(motion_free_file))
            os.symlink(motion_free_file, symlink_path)

            logging.info("[simulator::main] Finished. Processing took {:.2f} "
                         "seconds.".format(time.time() - start))


    logging.info("[simulator::main] Finished motion simulation for {} "
                 "set.".format(config_file["data_import"]["dataset_split"]))


if __name__ == "__main__":
    main()

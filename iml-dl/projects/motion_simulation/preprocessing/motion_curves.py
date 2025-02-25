"""
Preprocess the motion curves into adequate length and format to be loaded in
the motion simulation pipeline.
- run this script (from iml-dl/.) with:
    python -u ./projects/motion-simulation/preprocessing/motion_curves.py
        --input_dir folder_with_motion_data --output_dir output_folder
        --output_dir_selected output_folder_with_selected_curves
"""
import glob
import argparse
from utils import *


def main():
    parser = argparse.ArgumentParser(description="Process motion data")
    parser.add_argument("--input_dir", type=str,
                        default="/PATH/TO/ORIGINAL/MOTION/DATA/",
                        help="Path to the folder containing the text files"
                             "with the original motion data")
    parser.add_argument("--output_dir_all", type=str,
                        default="PATH/TO/PROCESSED/MOTION/DATA/",
                        help="Path to the folder where the preprocessed motion"
                             "data should be saved")
    parser.add_argument("--output_dir_selected", type=str,
                        default="/PATH/TO/PROCESSED/MOTION/DATA/SELECTED/",
                        help="Path to the folder where the selected "
                             "preprocessed motion data should be saved")
    args = parser.parse_args()
    os.makedirs(args.output_dir_all, exist_ok=True)
    os.makedirs(args.output_dir_selected, exist_ok=True)

    files = glob.glob(args.input_dir+'**.txt')

    curve_types = {
        '_dsc': {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [],
                 'r_z': [], 'filename': []},
        '-fMRI': {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [],
                  'r_z': [], 'filename': []},
        '_CVR': {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [],
                 'r_z': [], 'filename': []},
        '_NTT': {'t_x': [], 't_y': [], 't_z': [], 'r_x': [], 'r_y': [],
                 'r_z': [], 'filename': []},
    }

    repetition_times = {'_dsc': 1.3, '-fMRI': 1.5, '_CVR': 1.2, '_NTT': 1.0}

    curve_types = load_motion_data(files, curve_types)
    curve_types = convert_to_arrays(curve_types)
    print_curve_counts(curve_types)

    length_all = 150
    all_curves = get_equal_length_curves(curve_types, length_all,
                                         repetition_times)
    all_curves = correct_curves(all_curves)

    # calculate motion statistics:
    rms_displacement, motion_free, max_displacement = calculate_motion_statistics(
        all_curves, dset_shape=(12, 35, 92, 112), pixel_spacing=(3.3, 2, 2),
        radius=64., motion_free_threshold=2.0
    )
    print("RMS Displacement: {} +- {} mm".format(np.mean(rms_displacement),
                                              np.std(rms_displacement)))
    print("Motion Free Fraction (< 2.0 mm): {} +- {}".format(np.mean(motion_free),
                                         np.std(motion_free)))
    print("Max. Displacement: {} +- {} mm".format(np.mean(max_displacement),
                                               np.std(max_displacement)))

    if os.path.basename(args.input_dir[:-1]) == "Strong_Motion_AW":
        condition = motion_free > 0.45
        all_curves_sorted = {key: np.array(val)[condition]
                             for key, val in all_curves.items()}
        rms_displacement_sorted = rms_displacement[condition]
        max_displacement_sorted = max_displacement[condition]
        motion_free_sorted = motion_free[condition]

        print("Picking only curves with Motion Free Fraction >= {}:".format(
            np.min(motion_free_sorted))
        )
        print("Number of curves: ", len(motion_free_sorted))
        print("RMS Displacement: {} +- {} mm".format(
            np.mean(rms_displacement_sorted),
            np.std(rms_displacement_sorted))
        )
        print("Motion Free Fraction: {} +- {}".format(
            np.mean(motion_free_sorted),
            np.std(motion_free_sorted))
        )
        print("Max. Displacement: {} +- {} mm".format(
            np.mean(max_displacement_sorted),
            np.std(max_displacement_sorted))
        )
        save_curves(all_curves_sorted, rms_displacement_sorted,
                    motion_free_sorted, max_displacement_sorted,
                    output_dir=args.output_dir_selected,
                    dataset=os.path.basename(args.input_dir[:-1]).lower(),
                    repetition_times=repetition_times)
    else:
        save_curves(all_curves, rms_displacement, motion_free,
                    max_displacement,
                    output_dir=args.output_dir_selected,
                    dataset=os.path.basename(args.input_dir[:-1]).lower(),
                    repetition_times=repetition_times)

    save_curves(all_curves, rms_displacement, motion_free, max_displacement,
                output_dir=args.output_dir_all,
                dataset=os.path.basename(args.input_dir[:-1]).lower(),
                repetition_times=repetition_times)


if __name__ == "__main__":
    main()
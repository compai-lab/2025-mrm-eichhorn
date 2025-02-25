"""
This script is used to distribute the motion curves among different sets and
map between curves and subjects within the val and test splits.
- run this script (from iml-dl/.) with:
    python -u ./projects/motion-simulation/preprocessing/data_split_motion_curves.py
        --input_dir folder_with_motion_data --output_dir output_folder
        --new_train_test_split True or False
"""
import argparse
import datetime
import glob
import os
import json
from utils import *


def main():
    parser = argparse.ArgumentParser(description="Load and distribute curves"
                                                 "among different sets.")
    parser.add_argument("--input_dir", type=str,
                        default="/PATH/TO/SELECTED/MOTION/DATA/",
                        # insert path to motion data (selected and preprocessed curves)
                        help="Directory containing the processed motion curves"
                             "as JSON files.")
    parser.add_argument("--output_dir", type=str,
                        default="./projects/motion_simulation/configs/data_set_split/",
                        help="Where to store output.")
    parser.add_argument("--new_train_test_split", type=str,
                        default=False,
                        help="Whether a new train test split of the motion"
                             "curves should be performed or only a new mapping"
                             "of curves and subjects within the splits.")
    args = parser.parse_args()

    subjects = {
        "val": [ "SQ-struct-37", "SQ-struct-39", "SQ-struct-40"],
        "test": ["SQ-struct-33", "SQ-struct-38", "SQ-struct-43",
                 "SQ-struct-44", "SQ-struct-45", "SQ-struct-46",
                 "SQ-struct-47", "SQ-struct-48", "SQ-struct-00",
                 "SQ-struct-01", "SQ-struct-02", "SQ-struct-03",
                 "SQ-struct-04"],
    }

    if args.new_train_test_split:
        print("New train test split.")
        curves = load_curves(args.input_dir)
        sets = distribute_curves_into_sets(curves)

        current_date = datetime.datetime.now().strftime("%y-%m-%d")

        for set_name, curves in sets.items():
            with open(os.path.join(args.output_dir,
                                   f'{set_name}_filenames_{current_date}.txt'),
                      'w') as f:
                for curve in curves:
                    f.write(curve['filename'] + '\n')

    else:
        print("New mapping of curves and subjects within the splits.")
        for set_name in ['val', 'test']:
            files = sorted(glob.glob(os.path.join(args.output_dir,
                                           f'{set_name}_filenames*.txt')
                              ))
            most_recent_file = files[-1]
            curve_names = np.loadtxt(most_recent_file, dtype=str)

            num_curves_per_subject = len(curve_names) // len(subjects[set_name])
            distribution = {}

            for i, subject in enumerate(subjects[set_name]):
                start_index = i * num_curves_per_subject
                end_index = start_index + num_curves_per_subject
                distribution[subject] = curve_names[start_index:end_index].tolist()

            filename = most_recent_file.replace(
                ".txt", ".json"
            ).replace("_filenames_", "_distributed_filenames_")
            with open(filename, 'w') as f:
                json.dump(distribution, f)


if __name__ == "__main__":
    main()
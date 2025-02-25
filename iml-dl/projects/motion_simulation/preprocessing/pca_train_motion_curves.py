import os
import argparse
import numpy as np
from utils import *
import datetime


def main():
    parser = argparse.ArgumentParser(description="Load training curves and"
                                                 "perform PCA.")
    parser.add_argument("--input_dir", type=str,
                        default="/PATH/TO/SELECTED/PROCESSED/MOTION/DATA/",
                        help="Directory containing the processed motion curves"
                             "as JSON files.")
    parser.add_argument("--output_dir", type=str,
                        default="/PATH/TO/SELECTED/PROCESSED/MOTION/DATA"
                                "/PCA_Train_Curves/",
                        help="Where to store output.")
    parser.add_argument("--train_filenames", type=str,
                        default="./projects/motion_simulation/configs/"
                                "data_set_split/train_filenames_24-08-06.txt",
                        help="File containing the filenames of the training"
                             "curves.")
    parser.add_argument("--num_train_curves", type=int, default=90,
                        help="Number of training curves to generate.")
    parser.add_argument("--train_subjects", type=str, nargs='+',
                        default=["SQ-struct-30", "SQ-struct-31", "SQ-struct-32",
                                 "SQ-struct-34", "SQ-struct-41", "SQ-struct-42"],
                        help="List of training subjects.")
    args = parser.parse_args()


    os.makedirs(args.output_dir+"PCA_components/", exist_ok=True)
    os.makedirs(args.output_dir+"PCA_generated_curves/", exist_ok=True)


    if os.path.exists(args.output_dir+"PCA_components/mean_train.json"):
        print(f"The mean curve already exists at {args.output_dir}"
              f"PCA_components/. Exiting.")
    else:
        train_files = np.loadtxt(args.train_filenames, dtype=str)
        train_curves = load_curves(args.input_dir, items='motion_data',
                                   filenames=train_files)

        mean_train_curve = calculate_mean_curve(train_curves)
        save_single_curve(mean_train_curve, os.path.join(args.output_dir,
                                                         "PCA_components",
                                                         "mean_train.json"))
        for i, curve in enumerate(train_curves):
            train_curves[i] = subtract_curves(curve, mean_train_curve)

        pca_train, expl_var, expl_var_ratio = perform_pca(train_curves)
        for i, pca_component in enumerate(pca_train):
            save_single_curve(pca_component, os.path.join(args.output_dir,
                                                          "PCA_components",
                                                          f"pc_{i:02}.json"))
        np.savetxt(os.path.join(args.output_dir, "PCA_components",
                                "explained_variance.txt"),
                   np.concatenate(
                       ([np.arange(0, len(pca_train))], [expl_var]),
                       axis=0).T
                   )
        np.savetxt(os.path.join(args.output_dir, "PCA_components",
                                "explained_variance_ratio.txt"),
                   np.concatenate(
                       ([np.arange(0, len(pca_train))], [expl_var_ratio]),
                       axis=0).T
                   )

        # save pca-generated curves
        for i in range(args.num_train_curves):
            print(f"Generating curve {i+1}/{args.num_train_curves}...")
            motion_free = 0.99
            while motion_free > 0.93 or motion_free <= 0.45:
                pca_gen_curve = generate_pca_curve(pca_train, mean_train_curve,
                                                   expl_var)
                (rms_displacement, motion_free,
                 max_displacement) = calculate_motion_statistics(
                    pca_gen_curve, dset_shape=(12, 35, 92, 112),
                    pixel_spacing=(3.3, 2, 2), radius=64., motion_free_threshold=2.0
                )

            pca_gen_curve['RMS_displacement'] = rms_displacement.item()
            pca_gen_curve['motion_free'] = motion_free.item()
            pca_gen_curve['max_displacement'] = max_displacement.item()
            print("Motion-free fraction: {} - accepted.".format(motion_free))

            save_single_curve(pca_gen_curve, os.path.join(args.output_dir,
                                                          "PCA_generated_curves",
                                                          f"curve_{i:02}.json"))

        all_gen_curves = os.listdir(args.output_dir+"PCA_generated_curves/")
        num_curves_per_subject = len(all_gen_curves) // len(args.train_subjects)
        distribution = {}

        for i, subject in enumerate(args.train_subjects):
            start_index = i * num_curves_per_subject
            end_index = start_index + num_curves_per_subject
            distribution[subject] = all_gen_curves[start_index:end_index]

        filename = (f"./projects/motion_simulation/configs/data_set_split/"
                    f"train_distributed_filenames_"
                    f"{datetime.datetime.now().strftime('%y-%m-%d')}.json")
        with open(filename, 'w') as f:
            json.dump(distribution, f)


if __name__ == "__main__":
    main()

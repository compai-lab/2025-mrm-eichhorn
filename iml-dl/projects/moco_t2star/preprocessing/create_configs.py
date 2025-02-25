"""
create_configs.py
- code for creating a set of configuration files for optimising the MLP for
    several subjects
- run this script (from iml-dl/.) with:

    python -u ./projects/moco_t2star/preprocessing/create_configs.py
        --base_config_path /path_to_base_config_file.yaml
        --output_directory /path_to_output_directory/
        --set {train/val/test}
"""
import yaml
import os
import argparse
import glob
from datetime import datetime


def is_timestamp_valid(timestamp):
    try:
        datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
        return True
    except ValueError:
        return False


def create_configs(base_config_path, output_directory, subjects, data_set):
    """Create individual config files for each subject."""

    with open(base_config_path, 'r') as base_config_file:
        base_config = yaml.safe_load(base_config_file)
        checkpoint_path = os.path.dirname(os.path.abspath(
            base_config['trainer']['params']['checkpoint_path']
        ))

    for subject in subjects:
        individual_config = base_config.copy()
        individual_config['experiment']['group'] = individual_config['name']
        individual_config['name'] = (f"{individual_config['name']}"
                                     f"_sub-{subject}-{data_set}")
        individual_config['data_loader']['params']['args']['select_one_scan'] = subject
        if "finetune" in individual_config['trainer']['params'].keys():
            individual_config['trainer']['params']['finetune']['subject'] = subject

        timestamp = os.path.basename(output_directory).replace("individual_configs_", "")
        if not is_timestamp_valid(timestamp):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        individual_config['trainer']['params']['checkpoint_path'] = os.path.join(
            checkpoint_path, f"weights/group-{timestamp}/{subject}/"
        )
        individual_config['downstream_tasks']['T2StarMotionCorrection']['checkpoint_path'] = os.path.join(
            checkpoint_path, f"downstream_metrics/group-{timestamp}/{subject}/"
        )

        output_path = os.path.join(output_directory, f"config_{data_set}_{subject}.yaml")
        with open(output_path, 'w') as output_file:
            yaml.safe_dump(individual_config, output_file)


def get_subjects_for_set(links_to_data_path):
    """Return the subjects for the given set."""

    subjects = glob.glob(f'{links_to_data_path}/raw_data/**fV4.mat')
    subjects = sorted([os.path.basename(s).split("_nr")[0] for s in subjects])

    return subjects


def main():
    parser = argparse.ArgumentParser(description="Create individual config files")
    parser.add_argument(
        "--base_config_path", type=str,
        default='./projects/moco_t2star/configs/config_train_base.yaml',
        help="Path to the base config file"
    )
    parser.add_argument(
        "--output_directory", type=str,
        default='./projects/moco_t2star/configs/individual_configs',
        help="Output directory for the individual config files"
    )
    parser.add_argument(
        "--set", type=str, default='val',
        help="Set to create configs for (train/val/test)"
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.base_config_path, 'r'))
    links_to_data_path = config["data_loader"]["params"]["args"]["data_dir"]["train"]
    subjects = get_subjects_for_set(links_to_data_path)

    os.makedirs(args.output_directory, exist_ok=True)
    create_configs(args.base_config_path, args.output_directory, subjects,
                   args.set)

    print(f"Created individual config files for {len(subjects)} "
          f"subjects in {args.set} set.")


if __name__ == "__main__":
    main()

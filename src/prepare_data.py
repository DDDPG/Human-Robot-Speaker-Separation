import csv
import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def prepare_dataset(
    datapath,
    savepath,
    n_spks=2,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    random_seed=42,
    skip_prep=False,
    use_relative_paths=False,
):
    """
    Prepare dataset and split it into training, validation, and test sets according to specified ratios.

    Arguments:
    ----------
        datapath (str): Dataset path
        savepath (str): CSV file save path
        n_spks (int): Number of speakers
        train_ratio (float): Training set ratio (0-1)
        valid_ratio (float): Validation set ratio (0-1)
        test_ratio (float): Test set ratio (0-1)
        random_seed (int): Random seed
        skip_prep (bool): Whether to skip data preparation
    """
    if skip_prep:
        return
    
    # Ensure the sum of ratios is 1
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "The sum of ratios must equal 1"
    
    if "HRSP2mix" in datapath:
        # Read all metadata files
        metadata_list = []
        sample_paths = []
        idx = 0
        while True:
            sample_path = os.path.join(datapath, f"sample_{idx:04d}")
            metadata_path = os.path.join(sample_path, "metadata.json")
            if not os.path.exists(metadata_path):
                # print(f"metadata path: {(metadata_path)}")
                break
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                metadata_list.append(metadata)
                sample_paths.append(sample_path)
            idx += 1

        if len(metadata_list) == 0:
            raise ValueError("No metadata found")
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Split dataset
        train_valid_paths, test_paths = train_test_split(
            sample_paths,
            test_size=test_ratio,
            random_state=random_seed
        )
        
        train_paths, valid_paths = train_test_split(
            train_valid_paths,
            test_size=valid_ratio/(train_ratio + valid_ratio),
            random_state=random_seed
        )
        
        # Move files to corresponding directories
        path_sets = {
            "train": train_paths,
            "valid": valid_paths,
            "test": test_paths
        }
        
        # Move files and update metadata
        set_metadata = {"train": [], "valid": [], "test": []}
        for set_type, paths in path_sets.items():
            for path in paths:
                idx = sample_paths.index(path)
                set_metadata[set_type].append(metadata_list[idx])
                
                # # Move audio files
                # base_name = os.path.basename(path)
                # for src, dst in [
                #     (os.path.join(path, "mixture.wav"), 
                #      os.path.join(datapath, set_type, "mixture", f"{base_name}.wav")),
                #     (os.path.join(path, "source1.wav"),
                #      os.path.join(datapath, set_type, "source1", f"{base_name}.wav")),
                #     (os.path.join(path, "source2.wav"),
                #      os.path.join(datapath, set_type, "source2", f"{base_name}.wav"))
                # ]:
                #     if os.path.exists(src):
                #         os.system(f"cp {src} {dst}")
        
        # Create CSV files for each set
        for set_type in ["train", "valid", "test"]:
            create_custom_dataset(
                datapath,
                savepath,
                set_metadata[set_type],
                dataset_name="HRSP",
                set_types=[set_type],
                use_relative_paths=use_relative_paths
            )
    else:
        raise ValueError("Unknown dataset")


def create_custom_dataset(
    datapath,
    savepath,
    metadata_list,
    dataset_name="custom",
    set_types=["train", "valid", "test"],
    folder_names={
        "source1": "source1",
        "source2": "source2",
        "mixture": "mixture",
    },
    use_relative_paths=False,
):
    """
    This function creates the csv file for a custom source separation dataset
    """
    os.makedirs(savepath, exist_ok=True)

    for set_type in set_types:
        # Get audio paths directly from metadata
        mix_fl_paths = [os.path.join(os.getcwd() + meta["merged_audio_path"]) for meta in metadata_list]
        s1_fl_paths = [os.path.join(os.getcwd() + meta["human_audio_path"]) for meta in metadata_list]
        s2_fl_paths = [os.path.join(os.getcwd() + meta["robot_audio_path"]) for meta in metadata_list]

        # Ensure all paths exist
        for paths in [mix_fl_paths, s1_fl_paths, s2_fl_paths]:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Audio file not found: {path}")

        # Define basic CSV columns first
        base_csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "human_wav",
            "human_wav_format",
            "human_wav_opts",
            "robot_wav",
            "robot_wav_format",
            "robot_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        # Get all keys from metadata
        all_metadata_keys = set()
        for metadata in metadata_list:
            all_metadata_keys.update(metadata.keys())

        # Remove potentially duplicate keys
        if "background_noise_name" in all_metadata_keys:
            all_metadata_keys.remove("background_noise_name")

        # Merge column names
        csv_columns = base_csv_columns + list(all_metadata_keys)

        with open(
            os.path.join(savepath, dataset_name + "_" + set_type + ".csv"),
            "w",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, human_path, robot_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):
                metadata = metadata_list[i]
                background_noise_name = metadata.get("background_noise_name", "")
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": os.path.abspath(mix_path) if not use_relative_paths else os.path.relpath(mix_path, start=datapath),
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "human_wav": os.path.abspath(human_path) if not use_relative_paths else os.path.relpath(human_path, start=datapath),
                    "human_wav_format": "wav",
                    "human_wav_opts": None,
                    "robot_wav": os.path.abspath(robot_path) if not use_relative_paths else os.path.relpath(robot_path, start=datapath),
                    "robot_wav_format": "wav",
                    "robot_wav_opts": None,
                    "noise_wav": background_noise_name,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                }
                # Complete other fields from metadata
                for key in all_metadata_keys:
                    if key in metadata:
                        row[key] = metadata[key]
                writer.writerow(row)

if __name__ == "__main__":
    
    # Create parser and add arguments
    parser = argparse.ArgumentParser(description='Prepare dataset with path options')
    parser.add_argument('--path-type', choices=['abs', 'rel'], default='rel',
                       help='Use absolute (abs) or relative (rel) paths in CSV files')
    args = parser.parse_args()
    
    # Convert path type argument to boolean
    use_relative_paths = (args.path_type == 'rel')
    
    # set cwd to parent directory to this file, absolute path
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    data_configs = [
        {
            "datapath": os.path.join(os.getcwd(), "data/HRSP2mix_8k/raw/"),
            "savepath": os.path.join(os.getcwd(), "data/HRSP2mix_8k/processed_raw/")
        },
        {
            "datapath": os.path.join(os.getcwd(), "data/HRSP2mix_8k/clean/"),
            "savepath": os.path.join(os.getcwd(), "data/HRSP2mix_8k/processed_clean/")
        },
        {
            "datapath": os.path.join(os.getcwd(), "data/HRSP2mix_16k/raw/"),
            "savepath": os.path.join(os.getcwd(), "data/HRSP2mix_16k/processed_raw/")
        },
        {
            "datapath": os.path.join(os.getcwd(), "data/HRSP2mix_16k/clean/"),
            "savepath": os.path.join(os.getcwd(), "data/HRSP2mix_16k/processed_clean/")
        }
    ]

    for config in data_configs:
        if os.path.exists(config["datapath"]):
            print(f"Processing {config['datapath']}")
            prepare_dataset(
                datapath=config["datapath"],
                savepath=config["savepath"],
                n_spks=2,
                skip_prep=False,
                use_relative_paths=use_relative_paths
            )
        else:
            print(f"Directory not found: {config['datapath']}")
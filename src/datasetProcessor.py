import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import argparse
import numpy as np
from src.interactive_mode import ask_user_inputs
from src.rawDataProcessor import create_geometry, generate_projections, generate_angles, load_raw_data, create_mat_file
from src.pickleConfigManager import get_configuration, save_configurations, load_configurations
from src.trainingConfigGenerator import generate_training_configs

PROCESSED_DATA_DIR = "./pickle_data"


def save_processed_data(data, save_path):
    """Save the processed data into a pickle file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved processed data to {save_path}")


def create_pickle_file(mat_file, config, filename):
    """Generate projections and save as a .pickle file."""
    from scipy.io import loadmat

    data = loadmat(mat_file)["img"].astype(np.float32)
    geo = create_geometry(config)

    train_angles = generate_angles(config["totalAngle"], config["numTrain"], config["startAngle"], False)
    val_angles = generate_angles(config["totalAngle"], config["numVal"], config["startAngle"], True)

    train_projections = generate_projections(data, geo, train_angles, config["noise"])
    val_projections = generate_projections(data, geo, val_angles, config["noise"])

    processed_data = {
        "image": data,
        "train": {"projections": train_projections, "angles": train_angles},
        "val": {"projections": val_projections, "angles": val_angles},
    }

    save_path = os.path.join(PROCESSED_DATA_DIR, filename, "data.pickle")
    save_processed_data(processed_data, save_path)

    return save_path



def process_dataset(name, config, high_precision):
    """
    Process the dataset: create .mat and .pickle files, update configs, and generate training configs.
    """
    raw_path = os.path.join("./raw_data", config["raw_file"])

    mat_file = create_mat_file(raw_path, config, name)

    pickle_file = create_pickle_file(mat_file, config, name)

    voxel_dims = config["nVoxel"]
    processed_file_name = f"{name}_{config['numTrain']}_{voxel_dims[0]}x{voxel_dims[1]}x{voxel_dims[2]}_{config['numVal']}val.pickle"
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, name, processed_file_name)

    os.rename(pickle_file, processed_file_path)
    print(f"✅ Renamed pickle file to {processed_file_path}")

    configurations = load_configurations()
    configurations[name] = config
    save_configurations(configurations)

    # TODO: training config path usage
    training_config_paths, validation_results = generate_training_configs(
        name=name,
        voxel_dims=voxel_dims,
        num_train=config["numTrain"],
        num_val=config["numVal"],
        use_dynamic=config.get("autoAdjust", False),
        high_precision=high_precision,
    )

    for model, path, result in validation_results:
        print(f"🟢 Model: {model}, Config Path: {path}, Validation Result: {result}")



def main():
    """
    Main function to process datasets.
    """
    parser = argparse.ArgumentParser(description="Process dataset for training configurations")
    parser.add_argument("--name", help="Dataset name (e.g., toothfairycbct, teapot)")
    parser.add_argument("--high-precision", help="Enable high precision processing", action="store_true")
    args = parser.parse_args()

    if args.name:
        dataset_name = args.name
        config = get_configuration(dataset_name)
        print(f"ℹ️ Using configuration for dataset: {dataset_name}")
    else:
        user_inputs = ask_user_inputs()
        dataset_name = user_inputs["name"]
        config = get_configuration(dataset_name, user_inputs["config"])

    process_dataset(
        name=dataset_name,
        config=config,
        high_precision=args.high_precision,
    )


if __name__ == "__main__":
    main()

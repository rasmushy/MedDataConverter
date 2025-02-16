#trainingConfigGenerator.py
import os
import yaml
from src.modelConfigManager import fetch_model_configuration, get_model_list, validate_model_training_config

TRAINING_CONFIG_DIR = "./training_config"

def calculate_epochs(n_voxel, base_epochs=1500, scale_factor=0.01):
    """Calculate the number of epochs dynamically based on voxel size."""
    return int(base_epochs + scale_factor * n_voxel[2])

def calculate_n_rays(n_voxel, base_rays=1024, scale_factor=0.1):
    """Calculate the number of rays dynamically based on voxel size."""
    return int(base_rays + scale_factor * max(n_voxel))

def calculate_n_samples(n_voxel, base_samples=320, scale_factor=2):
    """Calculate the number of samples dynamically based on voxel size."""
    return int(base_samples + scale_factor * max(n_voxel))

def generate_dynamic_values(dataset_name, num_train, num_val, n_voxel, model_type, use_dynamic):
    """
    Generate dynamic configuration values based on voxel size and other parameters.
    """
    if not isinstance(n_voxel, (tuple, list)) or len(n_voxel) != 3:
        raise ValueError(f"Invalid n_voxel: {n_voxel}. Must be a tuple or list of length 3.")

    return {
        "train": {
            "epoch": calculate_epochs(n_voxel) if use_dynamic else None,
            "n_rays": calculate_n_rays(n_voxel) if use_dynamic else None,
            "lrate_step": calculate_epochs(n_voxel) // 2 if use_dynamic else None,
        },
        "render": {
            "n_samples": calculate_n_samples(n_voxel) if use_dynamic else None,
        },
        "exp": {
            "expname": f"{dataset_name}_{num_train}",
            "expdir": f"./logs/{model_type}/",
            "datadir": f"./pickle_data/{dataset_name}/{dataset_name}_{num_train}_{n_voxel[0]}x{n_voxel[1]}x{n_voxel[2]}_{num_val}val.pickle",
        },
    }

def merge_configurations(static_config, dynamic_values):
    """
    Merge static configuration with dynamic runtime adjustments.
    """
    for key, value in dynamic_values.items():
        if key in static_config and isinstance(static_config[key], dict):
            static_config[key].update({k: v for k, v in value.items() if v is not None})
        else:
            static_config[key] = value
    return static_config

def create_training_config(dataset_name, num_train, n_voxel, num_val, model_type, use_dynamic):
    """
    Create a complete training configuration by merging static and dynamic values.
    """
    
    static_config = fetch_model_configuration(model_type, n_voxel=n_voxel)
    dynamic_values = generate_dynamic_values(dataset_name, num_train, num_val, n_voxel, model_type, use_dynamic)
    final_config = merge_configurations(static_config, dynamic_values)
    validate_model_training_config(final_config)
    return final_config

def save_training_config(config, model_type, dataset_name, num_train, voxel_dims, num_val, high_precision):
    """
    Save the training configuration to a YAML file.
    """
    precision_suffix = "_highPrecision" if high_precision else ""
    config_file_name = f"{dataset_name}_{num_train}_{voxel_dims[0]}x{voxel_dims[1]}x{voxel_dims[2]}_{num_val}val_{model_type}{precision_suffix}.yaml"
    config_path = os.path.join(TRAINING_CONFIG_DIR, model_type, config_file_name)

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path

def generate_training_configs(name, voxel_dims, num_train, num_val, use_dynamic, high_precision):
    """
    Generate and validate training configurations for all available models.
    """
    available_models = get_model_list()
    if not available_models:
        raise ValueError("No available models found for training configuration generation.")

    training_config_paths = []
    validation_results = []

    for model_type in available_models:
        try:
            config = create_training_config(
                dataset_name=name,
                num_train=num_train,
                n_voxel=voxel_dims,
                num_val=num_val,
                model_type=model_type,
                use_dynamic=use_dynamic,
            )
            config_path = save_training_config(config, model_type, name, num_train, voxel_dims, num_val, high_precision)
            training_config_paths.append(config_path)
            validation_results.append((model_type, config_path, "Valid"))
        except Exception as e:
            validation_results.append((model_type, None, f"Invalid: {str(e)}"))

    return training_config_paths, validation_results


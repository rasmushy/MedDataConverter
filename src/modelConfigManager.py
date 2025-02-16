#modelConfigManager.py
import os
import yaml

MODEL_CONFIG_FILE_PATH = "./src/models.yaml"

def load_model_configurations(file_path=MODEL_CONFIG_FILE_PATH):
    """
    Load model configurations from the YAML file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model configuration file not found: {file_path}")
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_model_list(model_configs=None):
    """
    Retrieve a list of available model types for training.
    Args:
        model_configs (dict, optional): Loaded model configurations. If None, configurations will be loaded dynamically.
    Returns:
        list: List of model types.
    """
    if model_configs is None:
        model_configs = load_model_configurations()
    return list(model_configs.keys())


def calculate_dynamic_parameters(n_voxel):
    """
    Calculate dynamic parameters like window size and number of windows based on voxel dimensions.
    """
    if not n_voxel:
        return [4, 4], 32

    largest_dim = max(n_voxel)
    if largest_dim <= 128:
        return [16, 16], 4
    elif largest_dim <= 256:
        return [8, 8], 16
    elif largest_dim <= 512:
        return [4, 4], 32
    else:
        return [2, 2], 64

# TODO: This could have stategies somewhere placed instead of this multi if statement, but it will do for now
def add_model_specific_values(model_config, model_type, n_voxel=None):
    """
    Enrich the base model configuration with dynamic values based on voxel size and model type.
    Args:
        model_config (dict): The base configuration for the model.
        model_type (str): The type of the model (e.g., Lineformer, tensorf).
        n_voxel (tuple): Dimensions of the voxel grid.
    Returns:
        dict: The enriched model configuration.
    """
    if model_type == "Lineformer":
        window_size, window_num = calculate_dynamic_parameters(n_voxel)
        model_config["train"]["window_size"] = window_size
        model_config["train"]["window_num"] = window_num

    if model_type == "tensorf":
        model_config["encoder"]["encoding"] = "tensorf"
        model_config["encoder"]["num_levels"] = 256

    if n_voxel and "render" in model_config:
        model_config["render"]["n_samples"] = max(192, n_voxel[2] // 2)
        if model_type == "tensorf":
            model_config["render"]["n_fine"] = max(192, n_voxel[2] // 2)

    return model_config

def fetch_model_configuration(model_type, n_voxel=None, model_configs=None):
    """
    Retrieve and enrich the model-specific configuration.
    Args:
        model_type (str): The type of the model (e.g., Lineformer, tensorf).
        n_voxel (tuple): Dimensions of the voxel grid.
        model_configs (dict): Loaded model configurations.
    Returns:
        dict: The enriched model configuration.
    """
    if model_configs is None:
        model_configs = load_model_configurations()

    base_config = model_configs.get(model_type, {}).copy()

    if not base_config:
        raise ValueError(f"No configuration found for model type: {model_type}")

    return add_model_specific_values(base_config, model_type, n_voxel)

def validate_model_training_config(config):
    """
    Validate a model training configuration by ensuring required keys and value types are present.
    Args:
        config (dict): The training configuration to validate.
    """
    required_keys = ["exp", "train", "render", "log", "network", "encoder"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    train_config = config.get("train", {})
    if not isinstance(train_config.get("n_rays"), int):
        raise ValueError(f"train.n_rays must be an integer, got: {train_config.get('n_rays')}")
    if not isinstance(train_config.get("epoch"), int):
        raise ValueError(f"train.epoch must be an integer, got: {train_config.get('epoch')}")

    render_config = config.get("render", {})
    if not isinstance(render_config.get("n_samples"), int):
        raise ValueError(f"render.n_samples must be an integer, got: {render_config.get('n_samples')}")

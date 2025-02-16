#pickleConfigManager.py
import os
import yaml

CONFIG_FILE_PATH = "./src/pickle_dataset_configs.yml"

def load_configurations():
    """Load dataset configurations from the configuration file."""
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"{CONFIG_FILE_PATH} not found. Creating a new one...")
        return {}
    with open(CONFIG_FILE_PATH, "r") as file:
        return yaml.safe_load(file) or {}

def save_configurations(configurations):
    """Save dataset configurations back to the configuration file."""
    def ensure_serializable(obj):
        """Convert unsupported types (e.g., tuples) to serializable formats."""
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: ensure_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ensure_serializable(i) for i in obj]
        return obj

    configurations = ensure_serializable(configurations)
    os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
    with open(CONFIG_FILE_PATH, "w") as file:
        yaml.dump(configurations, file, default_flow_style=False)
    print(f"✅ Configurations saved to {CONFIG_FILE_PATH}")

def create_default_configuration(params):
    """
    Create a default configuration for a new dataset based on provided parameters.
    """
    if "raw_file" not in params:
        raise ValueError("Missing required parameter: `raw_file`")

    total_slices = params.get("nVoxel", [256, 256, 256])[2]

    train_ratio = params.get("trainRatio", 0.7)
    val_ratio = params.get("valRatio", 0.2)

    numTrain = int(total_slices * train_ratio) if params.get("autoAdjust", False) else params.get("numTrain", 50)
    numVal = int(total_slices * val_ratio) if params.get("autoAdjust", False) else params.get("numVal", 50)

    return {
        "DSD": params.get("DSD", 1500),
        "DSO": params.get("DSO", 1000),
        "accuracy": 0.5,
        "mode": "cone",
        "filter": None,
        "totalAngle": 180.0,
        "startAngle": 0.0,
        "randomAngle": False,
        "trainRatio": train_ratio,
        "valRatio": val_ratio,
        "autoAdjust": params.get("autoAdjust", False),
        "dtype": params.get("dtype"),
        "convert": True,
        "rescale_slope": 1.0,
        "rescale_intercept": 0.0,
        "normalize": True,
        "noise": 0,
        "offOrigin": [0, 0, 0],
        "offDetector": [0, 0],
        "numTest": "auto",
        "raw_file": params["raw_file"],
        "nDetector": params.get("nDetector", [512, 512]),
        "dDetector": params.get("dDetector", [1.0, 1.0]),
        "nVoxel": params.get("nVoxel", [256, 256, 256]),
        "dVoxel": params.get("dVoxel", [1.0, 1.0, 1.0]),
        "numTrain": numTrain,
        "numVal": numVal,
    }


def create_configuration(dataset_name, params):
    """
    Create a new configuration or retrieve an existing one, with an option to overwrite.
    """
    configurations = load_configurations()

    if dataset_name in configurations:
        print(f"⚠️ Configuration for '{dataset_name}' already exists.")
        overwrite = input(f"Overwrite existing configuration? (y/n) [default: n]: ").strip().lower()
        if overwrite != "y":
            print("✅ Using the existing configuration.")
            return configurations[dataset_name]

    new_config = create_default_configuration(params)
    configurations[dataset_name] = new_config
    save_configurations(configurations)
    print(f"✅ Created and saved new configuration for '{dataset_name}'")
    return new_config

def validate_configuration(config):
    """
    Validate a configuration to ensure it is complete and consistent.
    """
    required_keys = ["raw_file", "nVoxel", "DSD", "DSO", "trainRatio", "valRatio"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"❌ Missing required configuration key: {key}")

    raw_path = os.path.join("./raw_data", config["raw_file"])
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"❌ Raw file '{raw_path}' not found. Ensure it exists.")

def get_configuration(dataset_name, params=None):
    """
    Retrieve and validate a configuration for the given dataset.
    """
    configurations = load_configurations()
    
    if dataset_name in configurations:
        config = configurations[dataset_name]
        validate_configuration(config)

        if params is not None:
            config["autoAdjust"] = params.get("autoAdjust", False)
            configurations[dataset_name] = config
            save_configurations(configurations)

        return config

    if params is None:
        raise ValueError(f"❌ No configuration found for dataset '{dataset_name}'. Provide `params` to create one.")
    
    print(f"ℹ️ No existing configuration found for '{dataset_name}', creating a new one...")
    return create_configuration(dataset_name, params)

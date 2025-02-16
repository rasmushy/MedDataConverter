#interactive_mode.py
import os
from src.rawDataProcessor import parse_shape

RAW_DATA_DIR = "./raw_data"

def safe_input(prompt, default=None, cast_fn=None):
    """
    Safely get user input with a default and optional type casting.
    Args:
        prompt (str): The prompt to display to the user.
        default (any): The default value to use if the user enters nothing.
        cast_fn (callable): A function to cast the input (e.g., int, float).
    Returns:
        any: The user's input, or the default value if input is empty.
    """
    user_input = input(f"{prompt} [default: {default}]: ").strip()
    if not user_input:
        return default
    try:
        return cast_fn(user_input) if cast_fn else user_input
    except (ValueError, TypeError):
        print(f"Invalid input. Using default value: {default}")
        return default

def get_raw_file_and_shape():
    """
    Prompt the user to select a raw file and specify its shape and dtype.
    Returns:
        tuple: Selected raw file, parsed shape (if required), and dtype.
    """
    valid_extensions = (".mha", ".raw", ".nii", ".dcm")
    files = [file for file in os.listdir(RAW_DATA_DIR) if file.endswith(valid_extensions)]

    if not files:
        raise FileNotFoundError(
            f"No valid raw files found in {RAW_DATA_DIR}. Please ensure the directory contains files with extensions {', '.join(valid_extensions)}."
        )

    print("Available files:")
    for i, file in enumerate(files):
        print(f"{i + 1}: {file}")

    file_choice = safe_input("Select a file by number", default="1", cast_fn=int) - 1
    if file_choice < 0 or file_choice >= len(files):
        raise ValueError("Invalid file selection. Please choose a valid number from the list.")

    raw_file = files[file_choice]

    if raw_file.endswith(".mha"):
        print("Selected file is .mha. Shape and dtype will be extracted automatically from the file.")
        return raw_file, None, None

    shape = safe_input("Enter the shape of the raw data in 'x' format (e.g., '256x256x128')", cast_fn=parse_shape)
    supported_dtypes = ["uint8", "uint16", "int16", "float32", "float64"]
    dtype = safe_input(
        f"Enter the data type of the raw file (supported: {', '.join(supported_dtypes)})",
        default="float32",
    )
    if dtype not in supported_dtypes:
        raise ValueError(f"Unsupported data type: {dtype}. Supported types are: {', '.join(supported_dtypes)}")

    return raw_file, shape, dtype



def get_processing_preferences():
    """
    Gather processing preferences interactively from the user.
    Returns:
        dict: Preferences including dynamic configuration. (visualization and precision options)
    """
    return {
        # TODO: Add additional questions to actually do something
        # "visualize": safe_input("Do you want to visualize the slices? (y/n)", default="y").lower() != "n",
        # "high_precision": safe_input("Do you want high precision processing? (y/n)", default="n").lower() == "y",
        "use_dynamic": safe_input("Do you want to use dynamic configuration? (y/n)", default="y").lower() != "n",
    }

def validate_voxel_size(voxel_size_str):
    """
    Validate and parse the voxel size input. Ensure it contains exactly three components.
    Args:
        voxel_size_str (str): Input string for voxel size (e.g., '1.0x1.0x1.0').
    Returns:
        list: List of three float values representing voxel dimensions.
    """
    default_voxel = [1.0, 1.0, 1.0]
    try:
        voxel_parts = list(map(float, voxel_size_str.split('x')))
        if len(voxel_parts) == 1:
            voxel_parts = voxel_parts * 3
        elif len(voxel_parts) == 2:
            voxel_parts.append(default_voxel[2])
        elif len(voxel_parts) != 3:
            raise ValueError("Voxel size must have exactly three dimensions (e.g., '1.0x1.0x1.0').")
        return voxel_parts
    except ValueError:
        raise ValueError("Invalid voxel size format. Please use 'XxYxZ' format.")

def get_additional_configuration(use_dynamic):
    """
    Gather additional configuration details interactively from the user.
    Args:
        use_dynamic (bool): Whether dynamic configuration is enabled.
    Returns:
        dict: Additional configuration parameters.
    """
    print("Please provide additional configuration details:")
    dsd = safe_input("Enter Distance Source Detector (DSD) in mm", default=1500, cast_fn=float)
    dso = safe_input("Enter Distance Source Origin (DSO) in mm", default=1000, cast_fn=float)
    detector_dims = safe_input("Enter detector pixel count as 'XxY' (e.g., '512x512')", default="512x512")
    d_detector = list(map(float, safe_input("Enter detector pixel size as 'XxY' (e.g., '1.0x1.0')", default="1.0x1.0").split('x')))
    
    voxel_size_str = safe_input("Enter voxel size as 'XxYxZ' (e.g., '1.0x1.0x1.0')", default="1.0x1.0x1.0")
    d_voxel = validate_voxel_size(voxel_size_str)

    num_train = 50
    num_val = 50
    auto_adjust = True
    if not use_dynamic:
        auto_adjust = False
        num_train = safe_input("Enter the number of training samples", default=50, cast_fn=int)
        num_val = safe_input("Enter the number of validation samples", default=50, cast_fn=int)
        
        epoch = safe_input("Enter the number of epochs", default=1500, cast_fn=int)
        n_rays = safe_input("Enter the number of rays per batch", default=1024, cast_fn=int)
        lrate_step = safe_input("Enter the learning rate step interval", default=1500, cast_fn=int)
        n_samples = safe_input("Enter the number of samples per ray", default=256, cast_fn=int)
    else:
        epoch = None
        n_rays = None
        lrate_step = None
        n_samples = None

    return {
        "DSD": dsd,
        "DSO": dso,
        "dDetector": d_detector,
        "dVoxel": d_voxel,
        "nDetector": list(map(int, detector_dims.split('x'))),
        "numTrain": num_train,
        "numVal": num_val,
        "autoAdjust": auto_adjust,
        "epoch": epoch,
        "n_rays": n_rays,
        "lrate_step": lrate_step,
        "n_samples": n_samples,
    }



def ask_user_inputs():
    """
    Interactively gather all required inputs from the user.
    Returns:
        dict: Consolidated inputs including dataset details, preferences, and additional configurations.
    """
    print("=== Dataset Configuration ===")
    dataset_name = safe_input("Enter the dataset name (e.g., 'head', 'leg', 'toothfairycbct')", default="default_dataset").strip()
    raw_file, shape, dtype = get_raw_file_and_shape()

    print("\n=== Processing Preferences ===")
    preferences = get_processing_preferences()

    print("\n=== Additional Configuration ===")
    additional_config = get_additional_configuration(preferences["use_dynamic"])

    if shape:  
        additional_config["nVoxel"] = shape
    if dtype: 
        additional_config["dtype"] = dtype

    additional_config["raw_file"] = raw_file

    return {
        "name": dataset_name,
        "raw_file": raw_file,
        **preferences,
        "config": additional_config,
    }



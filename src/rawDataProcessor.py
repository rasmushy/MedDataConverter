#rawDataProcessor.py
import os
import numpy as np
import SimpleITK as sitk
import scipy.io as sio
from tigre import Ax
from tigre.utilities.geometry import Geometry
from tigre.utilities import CTnoise

RAW_DATA_DIR = "./raw_data"
MAT_DATA_DIR = "./mat_data"

def convert_to_attenuation(data, rescale_slope, rescale_intercept):
    """Convert CT Hounsfield units (HU) to attenuation."""
    HU = data * rescale_slope + rescale_intercept
    mu_water = 0.206
    mu_air = 0.0004
    return mu_water + (mu_water - mu_air) / 1000 * HU

def apply_noise(data, noise_level):
    """Apply Poisson and Gaussian noise to projections."""
    if noise_level > 0:
        print(f"Adding noise with Gaussian level: {noise_level}")
        return CTnoise.add(data, Gaussian=[0, noise_level])
    return data

def create_geometry(config):
    """Create a ConeBeam CT geometry from the configuration."""
    geo = Geometry()
    geo.DSD = config["DSD"] / 1000
    geo.DSO = config["DSO"] / 1000
    geo.nDetector = np.array(config["nDetector"])
    geo.dDetector = np.array(config["dDetector"]) / 1000
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array(config["nVoxel"][::-1])
    geo.dVoxel = np.array(config["dVoxel"][::-1]) / 1000
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.array(config["offOrigin"][::-1]) / 1000
    geo.offDetector = np.array(config["offDetector"][::-1]) / 1000
    geo.accuracy = config["accuracy"]
    return geo

def generate_projections(data, geo, angles, noise_level=0):
    """
    Generate projections for given geometry and angles.
    Args:
        data (np.ndarray): The 3D volume data in [z, y, x] order.
        geo (Geometry): Geometry configuration for projection.
        angles (np.ndarray): Projection angles in radians.
        noise_level (float): Noise level for projections.
    Returns:
        np.ndarray: Projections with optional noise.
    """
    # [z, y, x] order in configs
    if data.shape != tuple(geo.nVoxel):
        raise ValueError(f"Input data should be of shape geo.nVoxel: {geo.nVoxel} not:{data.shape}")
    projections = Ax(data, geo, angles)[:, ::-1, :]
    return apply_noise(projections, noise_level)

def generate_angles(total_angle, num_angles, start_angle=0, random=False):
    """
    Generate angles for projections.
    Args:
        total_angle (float): Total angle in degrees.
        num_angles (int): Number of angles to generate.
        start_angle (float): Starting angle in degrees.
        random (bool): Whether to generate random angles.
    Returns:
        np.ndarray: Array of angles in radians.
    """
    if random:
        angles = np.sort(np.random.rand(num_angles) * (total_angle * np.pi / 180)) + (start_angle * np.pi / 180)
    else:
        angles = np.linspace(start_angle * np.pi / 180, (start_angle + total_angle) * np.pi / 180, num_angles, endpoint=False)
    return angles

def parse_shape(shape_str):
    """Parse shape from 'x' formatted string."""
    try:
        return tuple(map(int, shape_str.split('x')))
    except ValueError:
        raise ValueError(f"Invalid shape format: {shape_str}. Use 'x' format (e.g., '256x256x128').")

def detect_dtype_mha(file_path):
    """Detect data type of an .mha file."""
    image = sitk.ReadImage(file_path)
    pixel_type = image.GetPixelIDTypeAsString().lower()
    dtype_mapping = {
        "8-bit unsigned integer": "uint8",
        "16-bit unsigned integer": "uint16",
        "32-bit float": "float32",
        "64-bit float": "float32",  # TODO: Support float64 when high precision is enabled
    }
    dtype = dtype_mapping.get(pixel_type)
    if not dtype:
        raise ValueError(f"Unsupported pixel type in .mha file: {image.GetPixelIDTypeAsString()}.")
    return dtype

def load_dicom_data(image_path):
    """
    Load DICOM images as a raw 3D volume.
    Args:
        image_path (str): Path to the folder containing DICOM files.
    Returns:
        np.ndarray: Normalized 3D volume (float32).
    """
    print(f"Loading DICOM images from {image_path}...")

    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(image_path)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # Shape: (z, y, x)

    data_float = np.float32(image_array) / image_array.max() # Normalize data

    print(f"DICOM data loaded. Shape: {data_float.shape}")
    return data_float

def load_raw_data(raw_path, config):
    """Load and reshape raw data from different file formats."""
    if raw_path.endswith(".mha"):
        image = sitk.ReadImage(raw_path)
        data = sitk.GetArrayFromImage(image)
        config["nVoxel"] = tuple(reversed(image.GetSize()))
    elif os.path.isdir(raw_path):
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(raw_path))
        image = reader.Execute()
        data = sitk.GetArrayFromImage(image)
        config["nVoxel"] = data.shape[::-1]
    else:
        shape = tuple(config["nVoxel"])
        dtype = config.get("dtype")
        if not dtype:
            raise ValueError("Missing dtype for .raw file.")
        data = np.fromfile(raw_path, dtype=dtype).reshape(shape)

    data = np.transpose(data, (2, 1, 0))  # Align axes for TIGRE
    if data.shape != tuple(config["nVoxel"][::-1]):
        raise ValueError(f"Data shape mismatch: {data.shape} vs {config['nVoxel']}.")
    return data


def create_mat_file(raw_path, config, filename):
    """Create a .mat file from raw data with a configurable name."""
    data = load_raw_data(raw_path, config)
    
    if config.get("convert", True):
        data = convert_to_attenuation(
            data, config["rescale_slope"], config["rescale_intercept"]
        )

    save_dir = os.path.join(MAT_DATA_DIR, filename)
    os.makedirs(save_dir, exist_ok=True)
    
    mat_file_path = os.path.join(save_dir, f"{filename}.mat")
    sio.savemat(mat_file_path, {"img": data})
    
    print(f"✅ Saved .mat file to {mat_file_path}")
    return mat_file_path

def process_raw_file(raw_path, config):
    """
    Process the raw file using the configuration and return processed data.
    Args:
        raw_path (str): Path to the raw file.
        config (dict): Configuration dictionary.
    Returns:
        dict: Processed data ready for training.
    """
    print(f"Processing file: {raw_path}")

    raw_data = load_raw_data(raw_path, config)

    if not config.get("high_precision", False):
        raw_data = raw_data.astype(np.float32)

    total_projections = config["nVoxel"][2]

    if config.get("autoAdjust", True):
        config["numTrain"] = int(config["trainRatio"] * total_projections)
        config["numVal"] = int(config["valRatio"] * total_projections)
        print(f"Dynamically calculated numTrain: {config['numTrain']}, numVal: {config['numVal']}")
    else:
        total_set = config["numTrain"] + config["numVal"]
        if total_set > total_projections:
            raise ValueError(
                f"The total number of training and validation samples ({total_set}) exceeds available projections ({total_projections})."
            )
        config["trainRatio"] = config["numTrain"] / total_projections
        config["valRatio"] = config["numVal"] / total_projections
        print(f"Manually set numTrain: {config['numTrain']}, numVal: {config['numVal']}")
        print(f"Calculated trainRatio: {config['trainRatio']:.4f}, valRatio: {config['valRatio']:.4f}")

    geo = create_geometry(config)

    train_angles = generate_angles(
        config["totalAngle"], config["numTrain"], config["startAngle"], config["randomAngle"]
    )
    # TODO: High-precision mode (use float64) has issue with tigre that wants to create projections in float32
    train_projections = generate_projections(raw_data, geo, train_angles, config["noise"])

    val_angles = generate_angles(
        config["totalAngle"], config["numVal"], config["startAngle"], True
    )
    val_projections = generate_projections(raw_data, geo, val_angles, config["noise"])

    processed_data = {
        "image": raw_data,
        "train": {
            "projections": train_projections,
            "angles": train_angles,
        },
        "val": {
            "projections": val_projections,
            "angles": val_angles,
        },
    }
    for key, value in config.items():
        if key not in processed_data:
            processed_data[key] = value

    return processed_data

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from image_processing import ImageProcessor

def load_config(config_path):
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration as a dictionary.
    """
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def merge_args_with_config(args, config):
    """Merge command-line arguments with configuration file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : dict
        Configuration dictionary from YAML.

    Returns
    -------
    dict
        Combined configuration.
    """
    merged = config.copy()
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value
    return merged

def analyze_image(image_path, params):
    """Analyze a single image to extract brightness and background information.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    params : dict
        Dictionary of processing parameters.

    Returns
    -------
    tuple
        Analysis results, contours, bright zones, and image array.
    """
    processor = ImageProcessor()
    filename = os.path.basename(image_path)
    img_array = processor.open_image(image_path)
    iso, exposure_time, date, time = processor.extract_exif_data(image_path, params.get("time_correction"))

    total_r, total_g, total_b, num_pixels = processor.sum_rgb_values(img_array)
    total_gray = processor.sum_grayscale_values(img_array)

    all_bright_zones, bright_mask, all_contours = processor.detect_bright_zone(
        img_array, params.get("threshold", 200), params.get("use_otsu", False), params.get("morph_close", False)
    )

    min_radius_threshold = params.get("min_radius_threshold", 150)
    bright_zones = []
    contours = []
    for i, ((x, y), radius) in enumerate(all_bright_zones):
        if radius >= min_radius_threshold:
            bright_zones.append(((x, y), radius))
            contours.append(all_contours[i])

    bright_r, bright_g, bright_b, n_bright_pixels = processor.sum_rgb_values(img_array, bright_mask)
    bright_gray = processor.sum_grayscale_values(img_array, bright_mask)

    if params.get("include_bg", False):
        bg_mean_r, bg_mean_g, bg_mean_b, bg_mean_gray = processor.compute_background_correction(
            img_array, bright_zones, bright_mask,
            params.get("bg_offset", 200),
            params.get("bg_width", 200), filename)

        if isinstance(bg_mean_r, float):
            bright_r -= bg_mean_r * n_bright_pixels
            bright_g -= bg_mean_g * n_bright_pixels
            bright_b -= bg_mean_b * n_bright_pixels
            bright_gray -= bg_mean_gray * n_bright_pixels
    else:
        bg_mean_r = bg_mean_g = bg_mean_b = bg_mean_gray = 'NA'

    flux_scale = float(exposure_time) if exposure_time not in ("Unknown", 0) else 1.0
    scale_factor = 1.0 / flux_scale if params.get("export_flux") else \
                   processor.estimate_gain_from_iso(iso) if params.get("export_electrons") else 1.0
    gain_used = processor.estimate_gain_from_iso(iso) if params.get("export_electrons") else 'NA'

    result = [
        filename, iso, exposure_time, date, time, gain_used,
        num_pixels, n_bright_pixels,
        bright_r * scale_factor, bright_g * scale_factor,
        bright_b * scale_factor, bright_gray * scale_factor
    ]

    if params.get("include_bg", False):
        result += [bg_mean_r, bg_mean_g, bg_mean_b, bg_mean_gray]

    return result, contours, bright_zones, img_array

def process_images_in_directory(directory, output_dir, params):
    """Process all supported images in a directory and extract photometric data.

    Parameters
    ----------
    directory : str
        Path to the input directory.
    output_dir : str
        Directory to save processed images and results.
    params : dict
        Processing configuration parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted image data.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    processor = ImageProcessor()

    for filename in os.listdir(directory):
        if filename.lower().endswith(("jpg", "jpeg", "png", "nef")):
            image_path = os.path.join(directory, filename)
            result, contours, bright_zones, img_array = analyze_image(image_path, params)

            all_data.append(tuple(result))

            output_image_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")
            processor.highlight_bright_zones(
                img_array, output_image_path, contours, bright_zones,
                params.get("include_bg", False),
                params.get("bg_offset", 200),
                params.get("bg_width", 200)
            )

    columns = [
        "filename", "iso", "exposure_time", "date", "time", "gain",
        "num_pixels", "n_bright_pixels",
        "R_sum", "G_sum", "B_sum", "W_sum"
    ]
    if params.get("include_bg", False):
        columns += ["bg_mean_R", "bg_mean_G", "bg_mean_B", "bg_mean_W"]

    phot_df = pd.DataFrame(all_data, columns=columns)
    for col in ["R_sum", "G_sum", "B_sum", "W_sum"]:
        phot_df[col + "_norm"] = phot_df[col] / phot_df[col].max()

    return phot_df

def plot_normalized_brightness(phot_df, plot_file):
    """Plot normalized brightness values for R, G, B channels.

    Parameters
    ----------
    phot_df : pd.DataFrame
        DataFrame containing image brightness data.
    plot_file : str
        Output path for the plot image.
    """
    plt.figure(figsize=(10, 6))
    phot_df["datetime"] = pd.to_datetime(phot_df["date"] + " " + phot_df["time"])
    plt.scatter(phot_df["datetime"], phot_df["B_sum_norm"], label="B", color="blue", s=15)
    plt.scatter(phot_df["datetime"], phot_df["G_sum_norm"], label="G", color="green", s=15)
    plt.scatter(phot_df["datetime"], phot_df["R_sum_norm"], label="R", color="red", s=15)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlabel("Time [UT]")
    plt.ylabel("Normalized flux")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.show()

def main():
    """Main entry point for command-line execution.

    Parses arguments and processes eclipse images based on a configuration file.

    Available Arguments
    -------------------
    directory : str
        Path to the directory containing image files.
    --config : str, optional
        Path to the YAML configuration file. Defaults to 'config.yaml'.
    --threshold : int, optional
        Brightness threshold value for zone detection.
    --use-otsu : bool, optional
        Apply Otsu's thresholding method.
    --morph-close : bool, optional
        Apply morphological closing to fill gaps in detected zones.
    --export-flux : bool, optional
        Normalize output using exposure time.
    --export-electrons : bool, optional
        Normalize output using estimated gain from ISO.
    --include-bg : bool, optional
        Compute and subtract background flux from the bright zones.
    --bg-offset : int, optional
        Offset distance from bright zone for background annulus.
    --bg-width : int, optional
        Width of the annular region for background sampling.
    --min-radius : int, optional
        Minimum radius required to qualify as a bright zone.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--use-otsu", action="store_true")
    parser.add_argument("--morph-close", action="store_true")
    parser.add_argument("--export-flux", action="store_true")
    parser.add_argument("--export-electrons", action="store_true")
    parser.add_argument("--include-bg", action="store_true")
    parser.add_argument("--bg-offset", type=int)
    parser.add_argument("--bg-width", type=int)
    parser.add_argument("--min-radius", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    params = merge_args_with_config(args, config)

    output_dir = os.path.join(args.directory, params.get("output_subdir", "processed"))
    output_txt = os.path.join(output_dir, params.get("output_txt", "image_data.txt"))
    plot_file = os.path.join(output_dir, params.get("plot_file", "flux.png"))

    phot_df = process_images_in_directory(args.directory, output_dir, params)
    phot_df.to_csv(output_txt, sep=" ", index=False, float_format="%.3f")
    plot_normalized_brightness(phot_df, plot_file)

if __name__ == "__main__":
    main()

# Simple Solar Photometry

[![Documentation Status](https://readthedocs.org/projects/your-project-name/badge/?version=latest)](https://ericbroens.github.io/SimpleSolarPhotometry/)
## Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
  - [Python Dependencies](#python-dependencies)
  - [System Dependencies (ExifTool)](#system-dependencies-exiftool)
- [Usage](#usage)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## About
**Simple Solar Photometry** is a command-line tool for straightforward extraction of 'photometric' data from solar (eclipse) images. It's designed to help extract brightness data from various image formats, including raw NEF files, and visualize changes over time.

## Features
- **Image Handling**: Supports common image formats (JPG, JPEG, PNG) and RAW NEF files.
- **EXIF Data Extraction**: Automatically extracts crucial metadata like ISO, exposure time, and date/time from images.
- **Solar Disk Detection**: Utilizes adaptive thresholding techniques to identify bright regions (e.g., the solar disk).
- **Photometric Analysis**: Calculates summed RGB and grayscale pixel values within detected regions.
- **Background Correction**: Offers optional annular sampling to estimate and subtract background brightness from the detected solar regions.
- **Normalization Options**: Supports normalizing output by exposure time (flux) or by estimated electrons based on ISO values.
- **Data Export**: Processes images in a directory and outputs photometric data into a structured text file.
- **Visualization**: Generates time-series plots of normalized brightness values using Matplotlib.
- **Configurable**: All processing parameters can be conveniently set via a YAML configuration file or overridden via command-line arguments.

## Installation

### Python Dependencies
The Python libraries required for this tool are listed in the `requirements.txt` file. You can install them using `pip`:

```bash
pip install -r requirements.txt
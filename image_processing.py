import cv2
import rawpy
import numpy as np
from PIL import Image, ImageOps
import subprocess
from datetime import datetime, timedelta

class ImageProcessor:
    def open_image(self, image_path):
        """Open an image file and convert it to a NumPy array.

        Parameters
        ----------
        image_path : str
            Path to the image file (supports NEF and standard formats).

        Returns
        -------
        np.ndarray
            Image data as a NumPy array.
        """
        if image_path.lower().endswith(".nef"):
            with rawpy.imread(image_path) as raw:
                return raw.postprocess(
                    output_bps=16,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_color=rawpy.ColorSpace.sRGB
                )
        else:
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            return np.array(pil_img)

    def extract_exif_data(self, image_path, time_correction_params=None):
        """Extract ISO, exposure time, and date/time from image EXIF data.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        time_correction_params : dict, optional
            Parameters to adjust EXIF timestamp (days, hours, minutes, seconds, sign).

        Returns
        -------
        tuple
            ISO, exposure time, date, time (as strings).
        """
        try:
            process = subprocess.Popen(
                ["exiftool", "-ISO", "-ExposureTime", "-DateTimeOriginal", "-T", "-n", image_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()

            if stderr:
                print(f"ExifTool error for {image_path}: {stderr.decode('utf-8')}")
                return "Unknown", "Unknown", "Unknown", "Unknown"

            iso, exposure_time, date_time_str = stdout.decode("utf-8").strip().split("\t")

            if date_time_str != "Unknown":
                original_dt = datetime.strptime(date_time_str, "%Y:%m:%d %H:%M:%S")
                if time_correction_params:
                    delta = timedelta(
                        days=time_correction_params.get("days", 0),
                        hours=time_correction_params.get("hours", 0),
                        minutes=time_correction_params.get("minutes", 0),
                        seconds=time_correction_params.get("seconds", 0),
                    )
                    sign = time_correction_params.get("sign", 1)
                    corrected_dt = original_dt + (sign * delta)
                else:
                    corrected_dt = original_dt

                date = corrected_dt.strftime("%Y-%m-%d")
                time = corrected_dt.strftime("%H:%M:%S")
            else:
                date, time = "Unknown", "Unknown"

            return iso, exposure_time, date, time

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return "Unknown", "Unknown", "Unknown", "Unknown"

    def sum_rgb_values(self, img_array, mask=None):
        """Sum RGB values of an image optionally using a mask.

        Parameters
        ----------
        img_array : np.ndarray
            Image array.
        mask : np.ndarray, optional
            Binary mask array.

        Returns
        -------
        tuple
            Sum of R, G, B channels and number of pixels considered.
        """
        if mask is not None:
            img_array = img_array * (mask[:, :, None] // 255)

        total_r = img_array[:, :, 0].sum()
        total_g = img_array[:, :, 1].sum()
        total_b = img_array[:, :, 2].sum()
        num_pixels = np.count_nonzero(mask) if mask is not None else img_array.shape[0] * img_array.shape[1]

        return total_r, total_g, total_b, num_pixels

    def sum_grayscale_values(self, img_array, mask=None):
        """Sum grayscale pixel values of an image, optionally using a mask.

        Parameters
        ----------
        img_array : np.ndarray
            Image array.
        mask : np.ndarray, optional
            Binary mask array.

        Returns
        -------
        int
            Sum of grayscale values.
        """
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array

        if mask is not None:
            img_gray = img_gray * (mask // 255)

        return img_gray.sum()

    def detect_bright_zone(self, img_array, threshold=200, use_otsu=False, apply_morph_close=False):
        """Detect bright zones in an image using thresholding techniques.

        Parameters
        ----------
        img_array : np.ndarray
            Image array.
        threshold : int, optional
            Threshold value for binarization.
        use_otsu : bool, optional
            Use Otsu's method if True.
        apply_morph_close : bool, optional
            Apply morphological closing to mask.

        Returns
        -------
        tuple
            List of detected zones, binary mask, and contours.
        """
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array

        if img_gray.dtype != np.uint8:
            img_gray = (img_gray / 256).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        if use_otsu:
            _, bright_mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            bright_mask = cv2.adaptiveThreshold(
                img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

        if apply_morph_close:
            kernel = np.ones((5, 5), np.uint8)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bright_zones = [cv2.minEnclosingCircle(cnt) for cnt in contours]

        return bright_zones, bright_mask, contours

    def compute_background_correction(self, img_array, bright_zones, bright_mask, bg_annulus_offset, bg_annulus_width, filename):
        """Compute the average background level in an annular region around detected bright zones.

        Parameters
        ----------
        img_array : np.ndarray
            Original image.
        bright_zones : list
            List of bright zones as (center, radius).
        bright_mask : np.ndarray
            Binary mask of detected bright zones.
        bg_annulus_offset : int
            Offset of the background annulus from the bright zone.
        bg_annulus_width : int
            Width of the annulus.
        filename : str
            Filename for logging.

        Returns
        -------
        tuple
            Mean background R, G, B, and grayscale values or 'NA'.
        """
        bg_mask = np.zeros_like(bright_mask, dtype=np.uint8)
        for (x, y), radius in bright_zones:
            outer_r = int(radius + bg_annulus_offset + bg_annulus_width)
            inner_r = int(radius + bg_annulus_offset)
            cv2.circle(bg_mask, (int(x), int(y)), outer_r, 255, -1)
            cv2.circle(bg_mask, (int(x), int(y)), inner_r, 0, -1)

        bg_pixels = np.count_nonzero(bg_mask)
        expected_area = sum(np.pi * ((int(r + bg_annulus_offset + bg_annulus_width))**2 - (int(r + bg_annulus_offset))**2)
                            for (_, _), r in bright_zones)

        if bg_pixels < 0.05 * expected_area:
            print(f"Warning: Annular background mask too small in {filename}.")

        if bg_pixels > 0:
            bg_r, bg_g, bg_b, _ = self.sum_rgb_values(img_array, bg_mask)
            bg_gray = self.sum_grayscale_values(img_array, bg_mask)
            return bg_r / bg_pixels, bg_g / bg_pixels, bg_b / bg_pixels, bg_gray / bg_pixels

        return 'NA', 'NA', 'NA', 'NA'

    def estimate_gain_from_iso(self, iso):
        """Estimate the gain (e-/ADU) based on ISO value.

        Parameters
        ----------
        iso : str
            ISO value from EXIF.

        Returns
        -------
        float
            Estimated gain in electrons per ADU.
        """
        try:
            iso_value = int(iso)
            return 50.0 / iso_value
        except:
            return 1.0

    def highlight_bright_zones(self, img_array, output_path, contours, bright_zones, include_bg, bg_annulus_offset=200, bg_annulus_width=200):
        """Draw contours and background annulus on an image.

        Parameters
        ----------
        img_array : np.ndarray
            Input image.
        output_path : str
            Path to save the output image.
        contours : list
            Contours of bright zones.
        bright_zones : list
            Bright zone info.
        include_bg : bool
            Whether to include background annuli.
        bg_annulus_offset : int
            Offset from bright zone for background.
        bg_annulus_width : int
            Width of the background annulus.
        """
        if img_array.dtype == np.uint16:
            img_to_draw = (img_array / 256).astype(np.uint8)
        else:
            img_to_draw = img_array.copy()

        if len(img_to_draw.shape) == 2:
            img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_GRAY2BGR)
        elif img_to_draw.shape[2] == 3:
            img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_RGB2BGR)
        elif img_to_draw.shape[2] == 4:
            img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_RGBA2BGR)

        cv2.drawContours(img_to_draw, contours, -1, (255, 150, 0), 3)

        if include_bg:
            for (x, y), radius in bright_zones:
                inner_r = int(radius + bg_annulus_offset)
                outer_r = int(radius + bg_annulus_offset + bg_annulus_width)
                cv2.circle(img_to_draw, (int(x), int(y)), inner_r, (0, 255, 0), 2)
                cv2.circle(img_to_draw, (int(x), int(y)), outer_r, (0, 255, 0), 2)

        cv2.imwrite(output_path, img_to_draw, [cv2.IMWRITE_JPEG_QUALITY, 95])

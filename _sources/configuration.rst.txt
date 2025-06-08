Configuration Parameters
------------------------

Below is a description of each parameter that can be set in the `config.yaml` file:

.. glossary::

    output_subdir
        :type: str
        :default: "processed"
         Directory name within the input image directory where processed images and
         results (like `image_data.txt` and `flux.png`) will be saved.
        
    output_txt
        :type: str
        :default: "image_data.txt"
         Filename for the CSV/text file containing the extracted photometric data.

    plot_file
        :type: str
        :default: "flux.png"
         Filename for the generated brightness vs. time plot.

    time_correction
        :type: dict
        :default: None
         Optional parameters to apply a correction to the EXIF timestamp of images.
         This is useful if your camera's clock is not perfectly synchronized.
         It should be a dictionary with keys: `days`, `hours`, `minutes`, `seconds`, `sign` (1 for add, -1 for subtract).
         Example: `{days: 0, hours: 0, minutes: 5, seconds: 0, sign: 1}` to add 5 minutes.

    threshold
        :type: int
        :default: 200
         Brightness threshold value used in bright zone detection (e.g., for the solar disk).
         Pixels above this value are considered part of the bright zone.

    use_otsu
        :type: bool
        :default: False
         If set to `True`, Otsu's thresholding method will be used for bright zone
         detection instead of adaptive Gaussian thresholding.

    morph_close
        :type: bool
        :default: False
         If set to `True`, morphological closing will be applied to the detected
         bright zone mask. This helps fill small gaps and smooth the mask.

    export_flux
        :type: bool
        :default: False
         If set to `True`, the output brightness values will be normalized by the
         exposure time of each image (flux = brightness / exposure_time).

    export_electrons
        :type: bool
        :default: False
         If set to `True`, the output brightness values will be normalized by an
         estimated gain value derived from the image's ISO setting. This aims
         to convert ADU (Analog-to-Digital Units) to electrons.

    include_bg
        :type: bool
        :default: False
         If set to `True`, the average background brightness will be estimated
         using an annular region around the detected bright zones and subtracted
         from the bright zone's total brightness.

    bg_offset
        :type: int
        :default: 200
         The offset distance (in pixels) from the outer edge of the detected bright
         zone to the inner edge of the background annulus. Only relevant if `include_bg` is True.

    bg_width
        :type: int
        :default: 200
         The width (in pixels) of the annular region used for background sampling.
         Only relevant if `include_bg` is True.

    min_radius
        :type: int
        :default: 150
         Minimum radius (in pixels) for a detected bright region to be considered
         a valid "bright zone" (e.g., the sun). Smaller regions will be ignored.
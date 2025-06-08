Command-Line Interface
======================

This tool processes eclipse image datasets using configurable image analysis routines.

.. program:: eclipse_processor

Usage
-----

.. code-block:: bash

   python SimSolPh.py [directory] [--options]

Arguments
---------

.. option:: directory

   Path to the directory containing images (JPG, PNG, NEF).

.. option:: --config <file>

   Path to YAML configuration file (default: config.yaml).

.. option:: --threshold <int>

   Brightness threshold value used in bright zone detection.

.. option:: --use-otsu

   Use Otsu's thresholding instead of adaptive Gaussian.

.. option:: --morph-close

   Apply morphological closing to mask artifacts.

.. option:: --export-flux

   Normalize brightness values using exposure time.

.. option:: --export-electrons

   Normalize brightness using ISO-based gain estimate.

.. option:: --include-bg

   Estimate and subtract background brightness using annular sampling.

.. option:: --bg-offset <int>

   Offset of the background annulus (default: 200).

.. option:: --bg-width <int>

   Width of the background annulus (default: 200).

.. option:: --min-radius <int>

   Minimum radius for a region to be considered a bright zone.

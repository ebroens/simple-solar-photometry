Installation
============

This tool requires `ExifTool <https://exiftool.org/>`_, a powerful command-line application for reading, writing, and editing meta information in a wide variety of files. It is a system dependency and needs to be installed separately from the Python packages.

**Python Dependencies:**

The Python dependencies are listed in `requirements.txt` and can be installed using pip:

.. code-block:: bash

    pip install -r requirements.txt
    pip install sphinx # optional, for building the documentation

**System Dependencies (ExifTool):**

Follow the instructions below for your operating system to install ExifTool:

Ubuntu/Debian
-------------
.. code-block:: bash

    sudo apt install libimage-exiftool-perl

macOS
-----
If you use `Homebrew <https://brew.sh/>`_, you can install ExifTool via:

.. code-block:: bash

    brew install exiftool

Windows
-------
Download the executable from the official ExifTool website:

- `ExifTool Website <https://exiftool.org/>`_

Follow the instructions on the website for installation. Typically, you'll download a `.zip` file, extract the `exiftool(-k).exe` executable, and rename it to `exiftool.exe`. You may then need to place this executable in a directory that is included in your system's `PATH` environment variable to run it from any command prompt.
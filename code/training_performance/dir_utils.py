"""
Author: Jess Breda
Date: July 12, 2023
Description: General python utilities that are not specific to datajoint or plotting
"""

from pathlib import Path

### create directories ###


def make_dirs(base_path, dir_names):
    """
    Create directories in base_path if they don't exist

    Parameters
    ----------
    base_path : str
        path to base directory
    dir_names : list
        list of directory names to create in base_path
    """
    for dir_name in dir_names:
        Path(base_path, dir_name).mkdir(parents=True, exist_ok=True)
    print("directories created")

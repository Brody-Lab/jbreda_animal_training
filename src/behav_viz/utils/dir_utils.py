"""
Author: Jess Breda
Date: July 12, 2023
Description: General python utilities that are not specific to datajoint or plotting
"""

from pathlib import Path
import platform
import os


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


### Paths ###
if platform.system() == "Windows":

    username = os.getlogin()
    ANIMAL_TABLE_PATH = Path(
        f"C:\\Users\\{username}\\github\\jbreda_animal_training\\data\\animals.xlsx"
    )
    DATA_PATH = Path(f"C:\\Users\\{username}\\github\\jbreda_animal_training\\data\\days_dfs")

    FIGURES_BASE_PATH = Path(
        f"C:\\Users\\{username}\\github\\jbreda_animal_training\\figures"
    )

else:
    ANIMAL_TABLE_PATH = Path(
        "/Users/jessbreda/Desktop/github/jbreda_animal_training/data/animals.xlsx"
    )
    DATA_PATH = Path(
        "/Users/jessbreda/Desktop/github/jbreda_animal_training/data/days_dfs"
    )
    FIGURES_BASE_PATH = Path(
        "/Users/jessbreda/Desktop/github/jbreda_animal_training/figures"
    )

def get_figures_path(species, cohort):

    return FIGURES_BASE_PATH / species / cohort

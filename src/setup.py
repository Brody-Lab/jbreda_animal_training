from setuptools import setup, find_packages
from pathlib import Path

# Path to the directory of setup.py
setup_dir = Path(__file__).parent

# Path to requirements.txt (two directories up)
requirements_path = setup_dir.parent / "requirements.txt"

# Read the contents of the requirements file
with open(requirements_path) as f:
    requirements = f.read().splitlines()

setup(
    name="behav_viz",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    description="Behavioral training visualization tools for the Brody lab",
    author="Jess Breda",
    author_email="jbreda@princeton.edu",
)

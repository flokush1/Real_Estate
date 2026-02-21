"""
This is a setup script for the real_estate project.
It includes necessary configurations and dependencies.
It is used by setup tools to package and distribute the project.
"""

from setuptools import setup, find_packages
from typing import List


def get_requirements() -> List[str]:
    """
    This function returns a list of required packages for the project.
    """
    requirement_lst: List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                requirements = line.strip()
                if (
                    requirements and requirements != "-e ."
                ):  # ignore empty lines and -e .(editable installs)
                    requirement_lst.append(requirements)

    except FileNotFoundError:
        print("requirements.txt file not found.")
    return requirement_lst


setup(
    name="real_estate",
    version="0.1.0",
    author="kush-Vikas",
    packages=find_packages(),
    install_requires=get_requirements(),
)

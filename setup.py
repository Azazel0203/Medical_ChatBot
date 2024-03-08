from setuptools import find_packages, setup

setup(
    name = "Medical ChatBot",
    version="0.0.1",
    author="Aadarsh Kumar Singh",
    author_email="aadarshkr.singh.cd.ece21@itbhu.ac.in",
    packages=find_packages(),
    # this function will look for the constructor file (__init__.py)
    # in every folder and install those local packages in my enviorment
    install_requires = []
)
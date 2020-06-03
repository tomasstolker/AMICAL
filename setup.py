#!/usr/bin/env python

from setuptools import find_packages, setup

project_name = "miamis"

setup(
    name=project_name,
    version=0.1,  # __import__(project_name).__version__,
    packages=find_packages(),
    author='Anthony Soulain',
    author_email='anthony.soulain@sydney.edu.au.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Professional Astronomers',
        'Topic :: High Angular Resolution Astronomy :: Interferometry',
        'Programming Language :: Python :: 3.7'
    ],
    install_requires=["matplotlib", "munch", "numpy",
                      "astropy", "scipy", "termcolor", "tqdm",
                      "uncertainties", "astroquery"],

)
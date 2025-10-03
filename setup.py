#!/usr/bin/env python3
"""
Setup script for OCT Deformation Tracking Toolkit

This allows the package to be installed via pip:
    pip install -e .  (for development/editable mode)
    pip install .     (for regular installation)

Author: Callum Brown
"""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "OCT Deformation Tracking Toolkit for analyzing optical coherence tomography images"

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        'numpy>=1.21.0',
        'opencv-contrib-python>=4.5.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'Pillow>=9.0.0'
    ]

setup(
    name='oct-deformation-toolkit',
    version='1.0.0',
    author='Callum Brown',
    author_email='',  # Add your email if desired
    description='Toolkit for analyzing OCT images with co-registered stress and attenuation data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/callumbrown01/thesis',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
        ],
    },
    entry_points={
        'console_scripts': [
            'oct-toolkit=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'oct_deformation_toolkit': ['*.py'],
    },
    zip_safe=False,
    keywords='oct optical-coherence-tomography image-registration optical-flow biomechanics',
    project_urls={
        'Bug Reports': 'https://github.com/callumbrown01/thesis/issues',
        'Source': 'https://github.com/callumbrown01/thesis',
    },
)

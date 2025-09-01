import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from pathlib import Path
import itertools
import csv
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import time

# Parameter grids for each algorithm
param_grids = {
    'Farneback': {
        'pyr_scale': [0.5],
        'levels': [3, 4],
        'winsize': [15, 20, 25],
        'iterations': [3],
        'poly_n': [5, 7],
        'poly_sigma': [1.1, 1.5],
        'flags': [0]  # Add required flags parameter
    },
    'Lucas-Kanade': {
        'win_size': [(15,15), (21,21)],
        'max_level': [3, 4],
        'criteria': [(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)]
    },
    'SimpleFlow': {
        'layers': [3, 4],
        'averaging_block_size': [2, 3],
        'max_flow': [4, 8]
    },
    'Speckle Tracking': {
        'block_size': [16, 24, 32],
        'search_area': [32, 48, 64],
        'overlap': [0.5, 0.75]
    },
    'DIS': {
        'setFinestScale': [1, 2],
        'setPatchSize': [8, 12],
        'setPatchStride': [4, 6],
        'setGradientDescentIterations': [12, 16],
        'setVariationalRefinementAlpha': [1.0],
        'setVariationalRefinementDelta': [0.5],
        'setVariationalRefinementGamma': [0.5],
        'setUseMeanNormalization': [True],
        'setUseSpatialPropagation': [True]
    },
    'DeepFlow': {
        'sigma': [0.6, 1.0],
        'alpha': [1.0, 1.5],
        'delta': [0.5, 1.0],
        'gamma': [0.5, 1.0]
    },
    'PCAFlow': {
        'n_components': [3, 5],
        'patch_size': [5, 7],
        'n_iter': [10, 15]
    },
    'TVL1': {
        'tau': [0.25, 0.5],
        'lambda_': [0.15, 0.3],
        'theta': [0.3, 0.5],
        'nscales': [3, 4],
        'warps': [3, 5]
    },
    'BlockMatching': {
        'block_size': [16, 24],
        'max_range': [32, 48],
        'prev_pyr_scale': [0.5, 0.7],
        'block_stride': [8, 12]
    }
}

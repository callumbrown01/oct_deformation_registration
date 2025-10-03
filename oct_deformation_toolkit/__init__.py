"""
OCT Deformation Tracking Toolkit

A comprehensive toolkit for analyzing optical coherence tomography (OCT) images
with co-registered stress and attenuation measurements during tissue deformation.
"""

__version__ = "1.0.0"
__author__ = "Callum Brown"

from .core.data_loader import OCTDataLoader
from .core.flow_computation import OpticalFlowEngine
from .core.image_processing import ImageProcessor
from .visualization.canvas_renderer import CanvasRenderer
from .visualization.plot_manager import PlotManager
from .analysis.metrics_calculator import MetricsCalculator

__all__ = [
    'OCTDataLoader',
    'OpticalFlowEngine',
    'ImageProcessor',
    'CanvasRenderer',
    'PlotManager',
    'MetricsCalculator',
]

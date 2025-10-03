#!/usr/bin/env python3
"""
OCT Deformation Tracking Toolkit - Main Application Entry Point

This file serves as the entry point for the OCT Deformation Tracking application.
It initializes the main GUI and provides the interactive interface for analyzing
optical coherence tomography images with co-registered stress and attenuation data.

Usage:
    python main.py

For now, this launches the existing final.py interface. The modular components
in oct_deformation_toolkit/ can be used for programmatic access and batch processing.

Author: Callum Brown
Version: 1.0.0
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append('opencv-contrib-python')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import scipy
    except ImportError:
        missing_deps.append('scipy')
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
    
    try:
        import PIL
    except ImportError:
        missing_deps.append('Pillow')
    
    try:
        from tracking_algorithms import TrackingAlgorithms
    except ImportError:
        missing_deps.append('tracking_algorithms.py (ensure it is in the same directory)')
    
    if missing_deps:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing Dependencies",
            f"The following dependencies are missing:\n\n" +
            "\n".join(f"  • {dep}" for dep in missing_deps) +
            f"\n\nPlease install them using:\n\n" +
            f"pip install -r requirements.txt"
        )
        return False
    
    return True


def main():
    """
    Main entry point for the application.
    
    This function checks dependencies and launches the main GUI.
    """
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Import the main application from final.py
        # Note: In future versions, this will be replaced with a refactored UI module
        from final import StressAttenuationExplorer
        
        # Create main window
        root = tk.Tk()
        root.title("OCT Deformation Tracking Toolkit v1.0.0")
        
        # Set minimum window size
        root.minsize(1200, 800)
        
        # Center window on screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = 1400
        window_height = 900
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create and pack the main application
        app = StressAttenuationExplorer(root)
        app.pack(fill='both', expand=True)
        
        # Start the event loop
        root.mainloop()
        
    except ImportError as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Import Error",
            f"Failed to import application components:\n\n{str(e)}\n\n" +
            f"Ensure all required files are in the same directory."
        )
        sys.exit(1)
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Application Error",
            f"An unexpected error occurred:\n\n{str(e)}\n\n" +
            f"Please check the console for more details."
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Verification script to check that all modules import correctly.

Run this to verify the package structure is working properly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_imports():
    """
    Test that all package modules can be imported.
    """
    print("=" * 60)
    print("OCT Deformation Toolkit - Import Verification")
    print("=" * 60)
    print()
    
    checks = []
    
    # Test package import
    print("Checking package import...")
    try:
        import oct_deformation_toolkit
        print("✓ oct_deformation_toolkit imported")
        checks.append(True)
    except Exception as e:
        print(f"✗ Failed to import package: {e}")
        checks.append(False)
    
    # Test main package components
    print("\nChecking main components...")
    try:
        from oct_deformation_toolkit import (
            OCTDataLoader,
            OpticalFlowEngine,
            ImageProcessor,
            MetricsCalculator,
            CanvasRenderer,
            PlotManager
        )
        print("✓ All main classes imported")
        checks.append(True)
    except Exception as e:
        print(f"✗ Failed to import main classes: {e}")
        checks.append(False)
    
    # Test core modules individually
    print("\nChecking core modules...")
    try:
        from oct_deformation_toolkit.core import data_loader
        print("✓ core.data_loader")
        checks.append(True)
    except Exception as e:
        print(f"✗ core.data_loader: {e}")
        checks.append(False)
    
    try:
        from oct_deformation_toolkit.core import flow_computation
        print("✓ core.flow_computation")
        checks.append(True)
    except Exception as e:
        print(f"✗ core.flow_computation: {e}")
        checks.append(False)
    
    try:
        from oct_deformation_toolkit.core import image_processing
        print("✓ core.image_processing")
        checks.append(True)
    except Exception as e:
        print(f"✗ core.image_processing: {e}")
        checks.append(False)
    
    # Test analysis module
    print("\nChecking analysis module...")
    try:
        from oct_deformation_toolkit.analysis import metrics_calculator
        print("✓ analysis.metrics_calculator")
        checks.append(True)
    except Exception as e:
        print(f"✗ analysis.metrics_calculator: {e}")
        checks.append(False)
    
    # Test visualization modules
    print("\nChecking visualization modules...")
    try:
        from oct_deformation_toolkit.visualization import canvas_renderer
        print("✓ visualization.canvas_renderer")
        checks.append(True)
    except Exception as e:
        print(f"✗ visualization.canvas_renderer: {e}")
        checks.append(False)
    
    try:
        from oct_deformation_toolkit.visualization import plot_manager
        print("✓ visualization.plot_manager")
        checks.append(True)
    except Exception as e:
        print(f"✗ visualization.plot_manager: {e}")
        checks.append(False)
    
    # Test utils
    print("\nChecking utils...")
    try:
        from oct_deformation_toolkit.utils import DataExporter
        print("✓ utils.DataExporter")
        checks.append(True)
    except Exception as e:
        print(f"✗ utils.DataExporter: {e}")
        checks.append(False)
    
    # Test tracking algorithms
    print("\nChecking external dependencies...")
    try:
        from tracking_algorithms import TrackingAlgorithms
        print("✓ tracking_algorithms")
        checks.append(True)
    except Exception as e:
        print(f"✗ tracking_algorithms: {e}")
        checks.append(False)
    
    # Summary
    print()
    print("=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ All imports successful! Package structure is correct.")
        return True
    else:
        print("✗ Some imports failed. Check error messages above.")
        return False


def check_dependencies():
    """
    Check if required dependencies are installed.
    """
    print("\n" + "=" * 60)
    print("Dependency Check")
    print("=" * 60)
    print()
    
    dependencies = [
        ('numpy', 'np'),
        ('cv2', None),
        ('scipy', None),
        ('matplotlib', 'plt'),
        ('PIL', None),
        ('tkinter', 'tk')
    ]
    
    checks = []
    for module_name, alias in dependencies:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"✓ {module_name}")
            checks.append(True)
        except ImportError:
            print(f"✗ {module_name} - NOT INSTALLED")
            checks.append(False)
    
    passed = sum(checks)
    total = len(checks)
    print(f"\n{passed}/{total} dependencies available")
    
    return passed == total


def main():
    """
    Run all verification checks.
    """
    import_success = check_imports()
    dep_success = check_dependencies()
    
    print("\n" + "=" * 60)
    if import_success and dep_success:
        print("✓ VERIFICATION COMPLETE - All checks passed!")
        print("\nYou can now:")
        print("  1. Run the application: python main.py")
        print("  2. Try the examples: python examples/usage_examples.py")
        print("  3. Install the package: pip install -e .")
    else:
        print("✗ VERIFICATION FAILED - See errors above")
        if not dep_success:
            print("\nInstall missing dependencies:")
            print("  pip install -r requirements.txt")
    print("=" * 60)
    print()
    
    return import_success and dep_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# OCT Deformation Tracking Toolkit

A comprehensive Python toolkit for analyzing optical coherence tomography (OCT) images with co-registered stress and attenuation measurements during tissue deformation. This tool enables researchers to study the biomechanical response of soft tissues under loading conditions.

## Features

- **Optical Flow Registration**: Multiple algorithms (DIS, Farneback, PCAFlow, DeepFlow, TVL1) for image co-registration
- **Multi-Modal Visualization**: Simultaneous display of OCT, stress, and attenuation data
- **Deformation Analysis**: Compute and visualize attenuation-stress response ratios
- **Interactive Analysis**: Draw regions of interest and track tissue response over time
- **Trajectory Visualization**: View pixel displacement fields with configurable overlays
- **Data Export**: Export co-registered images for external analysis
- **Histogram Analysis**: Statistical visualization of response distributions

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV with contrib modules
- MATLAB files (.mat) containing OCT, stress, and attenuation data

### Setup

1. Clone the repository:
```bash
git clone https://github.com/callumbrown01/thesis.git
cd thesis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure tracking_algorithms.py is in your Python path or project directory

## Quick Start

### Running the Application

```bash
python main.py
```

### Loading Data

1. Click "Load Sequence" button
2. Select a folder containing:
   - `*OCT*.mat` - OCT image files
   - `*Stress*.mat` - Stress measurement files
   - `*Att*.mat` - Attenuation coefficient files

### Basic Workflow

1. **Load Sequence**: Import your OCT data with stress and attenuation measurements
2. **Select Algorithm**: Choose an optical flow algorithm (default: DIS)
3. **Navigate Frames**: Use Prev/Next buttons to step through the sequence
4. **Enable Overlays**: Toggle stress, attenuation, or response ratio visualizations
5. **Draw Regions**: Click and drag on canvases to create up to 4 analysis regions
6. **View Trajectories**: See how stress/attenuation evolve in selected regions
7. **Export Data**: Save co-registered images for further analysis

## Project Structure

```
oct_deformation_toolkit/
├── __init__.py                 # Package initialization
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── data_loader.py         # OCT/stress/attenuation data loading
│   ├── flow_computation.py    # Optical flow algorithms
│   └── image_processing.py    # Image manipulation utilities
├── analysis/                   # Analysis tools
│   ├── __init__.py
│   └── metrics_calculator.py  # Quantitative metrics
├── visualization/              # Visualization components
│   ├── __init__.py
│   ├── canvas_renderer.py     # Tkinter canvas rendering
│   └── plot_manager.py        # Matplotlib plot management
└── ui/                        # User interface
    ├── __init__.py
    └── main_window.py         # Main application window

main.py                        # Application entry point
tracking_algorithms.py         # Optical flow implementations
requirements.txt               # Python dependencies
```

## Module Documentation

### Core Modules

#### `data_loader.py`
Handles loading and preprocessing of .mat files containing OCT images, stress measurements, and attenuation coefficients. Supports automatic colormap application and data validation.

**Key Class**: `OCTDataLoader`
- `load_sequence(folder_path)`: Load complete data sequence
- `get_frame(modality, index)`: Retrieve specific frame
- Automatic handling of different .mat file structures

#### `flow_computation.py`
Manages optical flow computation and composition for image registration.

**Key Class**: `OpticalFlowEngine`
- `precompute_flows(frames)`: Calculate all flow fields
- `warp_image(image, flow)`: Apply flow-based warping
- `get_cumulative_flow(index)`: Get displacement from reference

#### `image_processing.py`
Image manipulation utilities including filtering, masking, and conversions.

**Key Class**: `ImageProcessor`
- `block_average(data)`: Speckle noise reduction
- `create_adipose_mask(oct_image)`: Tissue segmentation
- `smooth_with_nan_handling(data)`: NaN-aware filtering

### Analysis Modules

#### `metrics_calculator.py`
Computes quantitative metrics for deformation analysis.

**Key Class**: `MetricsCalculator`
- `compute_response_ratio()`: ΔAttenuation/ΔStress calculation
- `compute_region_means()`: Regional statistics
- `compute_percentile_range()`: Robust value ranges

### Visualization Modules

#### `canvas_renderer.py`
Handles rendering on Tkinter canvas widgets.

**Key Class**: `CanvasRenderer`
- `draw_image_with_overlays()`: Main rendering method
- Trajectory visualization with automatic extent handling
- Scale bars and colorbars

#### `plot_manager.py`
Manages matplotlib plots for data visualization.

**Key Class**: `PlotManager`
- `plot_trajectories()`: Stress-attenuation scatter plots
- `save_histogram()`: Generate histogram figures

## Data Format

### Input Requirements

The toolkit expects .mat files with the following structure:

**OCT Files** (`*OCT*.mat`):
- Field name: 'img', 'OCT', 'oct', or 'image'
- Format: 2D array (H x W) or 3D array (H x W x 1)
- Type: uint8 grayscale or RGB

**Stress Files** (`*Stress*.mat`):
- Contains stress measurements in kPa
- Format: 2D (H x W) or 3D (H x W x N) for N frames
- Units: kilopascals (kPa)

**Attenuation Files** (`*Att*.mat`):
- Contains attenuation coefficients
- Format: 2D (H x W) or 3D (H x W x N) for N frames
- Units: mm⁻¹

### Coordinate System

- Origin: Top-left corner
- X-axis: Horizontal (left to right)
- Y-axis: Vertical (top to bottom)
- Flow vectors: (dx, dy) displacement in pixels

## Advanced Usage

### Custom Optical Flow Parameters

Modify algorithm parameters in `tracking_algorithms.py`:

```python
# Example: Adjust Farneback parameters
flow = cv2.calcOpticalFlowFarneback(
    prev, next, None,
    pyr_scale=0.5,    # Pyramid scale
    levels=3,         # Pyramid levels
    winsize=15,       # Window size
    iterations=3,     # Iterations
    poly_n=5,         # Polynomial expansion
    poly_sigma=1.2,   # Gaussian std
    flags=0
)
```

### Exporting Programmatically

```python
from oct_deformation_toolkit import OCTDataLoader, OpticalFlowEngine

# Load data
loader = OCTDataLoader()
loader.load_sequence('/path/to/data')

# Compute flows
flow_engine = OpticalFlowEngine(algorithm_name='DIS')
flow_engine.precompute_flows(loader.images['oct'])

# Export co-registered images
for i in range(loader.get_frame_count()):
    oct_frame = loader.get_frame('oct', i)
    flow = flow_engine.get_cumulative_flow(i)
    warped, mask = flow_engine.warp_image(oct_frame, flow)
    # Save warped...
```

### Batch Processing

```python
import glob
import os

# Process multiple sequences
sequences = glob.glob('/data/sequences/*/')

for seq_path in sequences:
    loader = OCTDataLoader()
    if loader.load_sequence(seq_path):
        # Analyze sequence...
        pass
```

## Troubleshooting

### Common Issues

**Issue**: "No OCT .mat files found"
- **Solution**: Ensure filenames contain "OCT" (case-insensitive)
- Check file permissions

**Issue**: Optical flow artifacts
- **Solution**: Try different algorithms (DIS usually most robust)
- Reduce noise in input images
- Check for extreme deformations

**Issue**: Memory errors with large sequences
- **Solution**: Process frames in batches
- Reduce image resolution if possible
- Clear flow cache periodically

**Issue**: Blank overlays
- **Solution**: Check stress/attenuation data ranges
- Verify data is not all NaN
- Ensure proper data alignment

## Performance Optimization

- **Algorithm Selection**: DIS is fastest and most accurate for OCT
- **Image Size**: Consider downsampling very large images
- **Caching**: Pre-computed flows are cached automatically
- **Visualization**: Reduce trajectory step size for faster rendering

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{brown2025oct,
  author = {Brown, Callum},
  title = {OCT Deformation Tracking Toolkit},
  year = {2025},
  url = {https://github.com/callumbrown01/thesis}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- OpenCV community for optical flow implementations
- SciPy developers for scientific computing tools
- Research collaborators and supervisors

## Contact

Callum Brown
- GitHub: [@callumbrown01](https://github.com/callumbrown01)
- Email: [Contact through GitHub]

## Version History

### v1.0.0 (2025)
- Initial release
- Multiple optical flow algorithms
- Interactive visualization
- Region-based analysis
- Co-registered image export

## Known Limitations

- Currently supports only .mat file format input
- Maximum 4 simultaneous analysis regions
- Requires MATLAB-compatible scipy.io for data loading
- Optical flow assumes relatively small inter-frame displacements
- Memory usage scales with sequence length and image size

## Future Development

- [ ] Support for DICOM and other medical imaging formats
- [ ] GPU acceleration for optical flow
- [ ] Automated quality metrics for registration
- [ ] 3D volume analysis capabilities
- [ ] Machine learning-based tissue classification
- [ ] Real-time processing pipeline
- [ ] Cloud deployment options

## References

1. Farnebäck, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion.
2. Kroeger, T., et al. (2016). Fast Optical Flow using Dense Inverse Search (DIS).
3. Zach, C., et al. (2007). A Duality Based Approach for Realtime TV-L1 Optical Flow.

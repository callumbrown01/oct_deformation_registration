# OCT Deformation Toolkit - Quick Start Guide

## How to Run the Application

### Method 1: Using Conda (If you have conda environment)
```powershell
conda run --name oct_tracker python main.py
```

### Method 2: Direct Python
```powershell
python main.py
```

### Method 3: Python 3
```powershell
python3 main.py
```

---

## First Time Setup

### 1. Navigate to Project Directory
```powershell
cd C:\Users\calsp\OneDrive\Documents\GitHub\thesis
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

For conda users:
```powershell
conda install opencv numpy scipy matplotlib pillow
pip install opencv-contrib-python
```

### 3. Verify Installation
```powershell
python verify_package.py
```

You should see: `✓ All imports successful!`

---

## Using the GUI

### Step-by-Step Workflow

1. **Launch Application**
   ```powershell
   python main.py
   ```

2. **Load Data**
   - Click "Select Sequence Folder" button
   - Choose folder containing:
     - `*OCT*.mat` files
     - `*Stress*.mat` files
     - `*Att*.mat` files

3. **Select Algorithm**
   - Choose from dropdown (recommend: **DIS**)

4. **Navigate Frames**
   - Use Prev/Next buttons

5. **Enable Overlays**
   - Toggle stress, attenuation, or response overlays

6. **Draw Analysis Regions**
   - Click and drag on canvases (up to 4 regions)

7. **Export Results**
   - Click "Export Data" button

---

## Installation (From GitHub)

```bash
# Clone repository
git clone https://github.com/callumbrown01/thesis.git
cd thesis

# Install dependencies
pip install -r requirements.txt

# Optional: Install as package
pip install -e .
```

---

## Programmatic Usage

### Programmatic Usage

```python
from oct_deformation_toolkit import (
    OCTDataLoader, OpticalFlowEngine, MetricsCalculator
)

# Load data
loader = OCTDataLoader()
loader.load_sequence('/path/to/data')

# Compute flows
flow_engine = OpticalFlowEngine(algorithm_name='DIS')
oct_frames = [loader.get_frame('oct', i) for i in range(loader.get_frame_count())]
flow_engine.precompute_flows(oct_frames)

# Analyze
metrics = MetricsCalculator()
response = metrics.compute_response_ratio(atten_ref, atten_curr, stress_ref, stress_curr)
```

## Common Tasks

### Load Sequence
```python
loader = OCTDataLoader()
success = loader.load_sequence('/path/to/folder')
```

### Get Frame Data
```python
oct_img = loader.get_frame('oct', frame_index)
stress_map = loader.get_frame('stress', frame_index)
atten_map = loader.get_frame('attenuation', frame_index)
```

### Compute Optical Flow
```python
flow_engine = OpticalFlowEngine(algorithm_name='DIS')
flow_engine.precompute_flows(oct_frames)
cumulative_flow = flow_engine.get_cumulative_flow(frame_index)
```

### Warp Image
```python
warped_img, valid_mask = flow_engine.warp_image(image, flow)
```

### Apply Smoothing
```python
processor = ImageProcessor()
smoothed = processor.smooth_with_nan_handling(data, sigma=2.0)
```

### Compute Response Ratio
```python
metrics = MetricsCalculator()
response = metrics.compute_response_ratio(
    atten_reference, atten_current,
    stress_reference, stress_current
)
```

### Export Data
```python
from oct_deformation_toolkit.utils import DataExporter

exporter = DataExporter(output_dir='./exports')
exporter.export_coregistered_sequence(oct_frames, stress_frames, atten_frames)
```

## Available Algorithms

- **DIS** (Dense Inverse Search) - Recommended
- **Farneback** - Fast, good for small displacements
- **PCAFlow** - Principal component analysis based
- **DeepFlow** - Deep matching
- **TVL1** - Total variation L1 optical flow

## File Structure Requirements

```
data_folder/
├── sequence_OCT.mat      (or *OCT*.mat)
├── sequence_Stress.mat   (or *Stress*.mat)
└── sequence_Att.mat      (or *Att*.mat)
```

## Key Classes

| Class | Purpose | Module |
|-------|---------|--------|
| `OCTDataLoader` | Load .mat files | `core.data_loader` |
| `OpticalFlowEngine` | Compute/compose flows | `core.flow_computation` |
| `ImageProcessor` | Image manipulation | `core.image_processing` |
| `MetricsCalculator` | Compute metrics | `analysis.metrics_calculator` |
| `CanvasRenderer` | Tkinter rendering | `visualization.canvas_renderer` |
| `PlotManager` | Matplotlib plots | `visualization.plot_manager` |
| `DataExporter` | Export results | `utils` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No OCT files found" | Ensure filenames contain "OCT" |
| Blank overlays | Check data ranges, verify alignment |
| Memory errors | Process in batches, reduce resolution |
| Flow artifacts | Try different algorithm (DIS recommended) |

## Keyboard Shortcuts (GUI)

- **Left Arrow**: Previous frame
- **Right Arrow**: Next frame
- **Escape**: Clear selections

## Default Settings

- **Flow Algorithm**: DIS
- **Colormap (Stress)**: Jet with log scale
- **Colormap (Attenuation)**: Viridis
- **Colormap (Response)**: Plasma
- **Overlay Alpha**: 0.5
- **Trajectory Step**: 5 pixels

## Units

- **Stress**: kPa (kilopascals)
- **Attenuation**: mm⁻¹ (inverse millimeters)
- **Response Ratio**: mm⁻¹/kPa
- **Displacement**: pixels

## Tips

1. **DIS algorithm** is most robust for OCT data
2. **Block averaging** reduces speckle noise
3. **Log scale** for stress reveals low-value details
4. **Pre-compute flows** for faster frame navigation
5. **Export 16-bit TIFF** to preserve precision
6. **Use ROIs** to track specific tissue regions

## Getting Help

- Check README.md for detailed documentation
- See examples/usage_examples.py for code samples
- Open issue on GitHub for bugs/questions
- Read CONTRIBUTING.md for development guidelines

# Contributing to OCT Deformation Toolkit

Thank you for your interest in contributing to the OCT Deformation Toolkit! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/thesis.git
   cd thesis
   ```
3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install in development mode**:
   ```bash
   pip install -e .
   pip install -e ".[dev]"  # Include development dependencies
   ```

## Development Workflow

### 1. Create a Branch

Create a new branch for your feature or bugfix:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

Use descriptive branch names:
- `feature/add-new-algorithm`
- `bugfix/fix-memory-leak`
- `docs/improve-readme`
- `refactor/optimize-flow-computation`

### 2. Make Your Changes

- Write clear, concise commit messages
- Follow the existing code style and conventions
- Add docstrings to new functions/classes
- Include type hints where appropriate
- Update documentation as needed

### 3. Test Your Changes

Before submitting, ensure:
- Your code runs without errors
- Existing functionality is not broken
- New features work as expected
- Code follows Python best practices

### 4. Submit a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Go to the original repository on GitHub
3. Click "New Pull Request"
4. Select your branch and provide a clear description

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation

- All public functions/classes should have docstrings
- Use Google-style docstrings format:
  ```python
  def example_function(param1: int, param2: str) -> bool:
      """
      Brief description of what the function does.
      
      Args:
          param1: Description of first parameter
          param2: Description of second parameter
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When invalid input is provided
      """
      pass
  ```

### Type Hints

Use type hints for function signatures:
```python
from typing import List, Optional, Tuple

def process_frames(
    frames: List[np.ndarray],
    algorithm: str = 'DIS'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Process image frames."""
    pass
```

## Project Structure

```
oct_deformation_toolkit/
├── __init__.py              # Package initialization
├── core/                    # Core functionality
│   ├── data_loader.py       # Data loading utilities
│   ├── flow_computation.py  # Optical flow algorithms
│   └── image_processing.py  # Image manipulation
├── analysis/                # Analysis tools
│   └── metrics_calculator.py
├── visualization/           # Visualization components
│   ├── canvas_renderer.py
│   └── plot_manager.py
└── utils/                   # Utility functions
    └── __init__.py          # Export utilities
```

## Adding New Features

### Adding a New Optical Flow Algorithm

1. Add implementation to `tracking_algorithms.py`
2. Update `OpticalFlowEngine` in `core/flow_computation.py`
3. Add algorithm to dropdown in UI
4. Document parameters and usage
5. Test with sample data

### Adding New Metrics

1. Add calculation method to `analysis/metrics_calculator.py`
2. Follow existing patterns (handle NaN values, validate inputs)
3. Add comprehensive docstring
4. Include example usage in documentation

### Adding New Visualization

1. Add method to appropriate visualization class
2. Support both interactive (matplotlib) and static output
3. Include colormap/scaling options
4. Document parameters clearly

## Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Minimal code to reproduce the problem
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: 
   - Python version
   - OS (Windows/Linux/macOS)
   - Package versions (`pip freeze`)
6. **Error messages**: Full traceback if applicable

Example:
```markdown
### Bug Description
Memory leak when processing large sequences

### Steps to Reproduce
1. Load sequence with 1000+ frames
2. Compute flows using DIS algorithm
3. Monitor memory usage - continues to grow

### Expected
Memory should stabilize after flow computation

### Actual
Memory grows continuously, eventually crashes

### Environment
- Python 3.10.5
- Windows 11
- opencv-contrib-python==4.6.0.66
```

## Feature Requests

When requesting features, please include:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches you've considered
4. **Additional context**: Screenshots, examples, etc.

## Testing

Currently, the project uses manual testing. Contributions to add automated tests are welcome!

### Manual Testing Checklist

Before submitting a PR, test:

- [ ] Data loading with various .mat file formats
- [ ] Optical flow computation with different algorithms
- [ ] Co-registration accuracy
- [ ] Metrics calculation correctness
- [ ] UI responsiveness
- [ ] Export functionality
- [ ] Cross-platform compatibility (if possible)

## Documentation

When adding features, update:

1. **README.md**: If it affects basic usage
2. **Docstrings**: For all new code
3. **Examples**: Add usage examples if applicable
4. **CHANGELOG**: Document significant changes

## Questions?

If you have questions:

1. Check existing [Issues](https://github.com/callumbrown01/thesis/issues)
2. Open a new issue with the "question" label
3. Be specific and provide context

## Code of Conduct

- Be respectful and professional
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume positive intent

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- README.md (for significant contributions)
- Commit history

Thank you for contributing! 🎉

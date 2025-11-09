# Fixing NumPy 2.x and PyTorch Compatibility Issues

## Problem Description

When working with AI/ML projects from this repository, you may encounter the following error:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
RuntimeError: Numpy is not available
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

This occurs because:
- Many PyTorch versions were compiled against NumPy 1.x
- NumPy 2.x introduced breaking changes in the C API
- The binary incompatibility causes PyTorch to fail when importing

## Solutions

### Solution 1: Downgrade NumPy (Recommended for Quick Fix)

The fastest solution is to downgrade to NumPy 1.x:

```bash
pip uninstall numpy
pip install "numpy<2.0"
```

Or specify an exact compatible version:

```bash
pip install numpy==1.26.4
```

### Solution 2: Upgrade PyTorch (Recommended for Long-term)

Upgrade to PyTorch 2.1.0 or later, which has better NumPy 2.x support:

```bash
# For CPU only
pip install --upgrade torch torchvision torchaudio

# For CUDA 11.8
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Solution 3: Create a Compatible Environment

Create a new virtual environment with compatible versions:

```bash
# Using conda
conda create -n ml_env python=3.10
conda activate ml_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy=1.26

# Using venv
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
pip install torch torchvision torchaudio
pip install "numpy<2.0"
```

### Solution 4: Pin Dependencies in requirements.txt

Create or update your `requirements.txt` with compatible versions:

```txt
numpy>=1.21.0,<2.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
scikit-image>=0.21.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Verification

### Quick Manual Test

After applying a solution, verify the installation:

```python
import numpy as np
import torch

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test basic functionality
x = np.array([1, 2, 3])
tensor = torch.from_numpy(x)
print(f"NumPy array converted to tensor: {tensor}")
```

Expected output should show:
- NumPy version 1.x.x (or 2.x.x with PyTorch 2.1+)
- PyTorch version without errors
- Successful tensor conversion

### Automated Verification Script

For a comprehensive check, use our verification script:

```bash
python verify_environment.py
```

This script will:
- Check all installed ML packages and versions
- Test NumPy/PyTorch compatibility
- Verify basic tensor operations
- Test CUDA availability
- Provide specific recommendations if issues are found

## Version Compatibility Matrix

| PyTorch Version | NumPy Version | Status |
|----------------|---------------|--------|
| <= 2.0.1 | 1.21 - 1.26 | ✅ Recommended |
| <= 2.0.1 | 2.0+ | ❌ Not Compatible |
| >= 2.1.0 | 1.21 - 1.26 | ✅ Recommended |
| >= 2.1.0 | 2.0+ | ⚠️ Experimental Support |
| >= 2.2.0 | 2.0+ | ✅ Better Support |

## Common Related Issues

### Issue: cv2 (OpenCV) compatibility

```bash
pip install opencv-python-headless==4.8.1.78
```

### Issue: scikit-image compatibility

```bash
pip install scikit-image==0.21.0
```

### Issue: Multiple packages requiring different NumPy versions

Use a virtual environment and install packages in the correct order:

```bash
python -m venv env
source env/bin/activate
pip install "numpy<2.0"
pip install torch torchvision
pip install opencv-python scikit-image scipy pandas
```

## Docker Solution

For maximum compatibility, use Docker with a proven environment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    opencv-python==4.8.1.78 \
    scikit-image==0.21.0

COPY . /app

CMD ["python", "your_script.py"]
```

Build and run:

```bash
docker build -t ml-project .
docker run --gpus all ml-project
```

## Prevention for Future Projects

1. **Always specify version constraints** in `requirements.txt`
2. **Use virtual environments** for each project
3. **Document your environment**: `pip freeze > requirements.txt`
4. **Test after updates**: Run your test suite after updating dependencies
5. **Use Docker** for production deployments

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues)

## Contributing

If you found another solution or have improvements to this guide, please submit a pull request!

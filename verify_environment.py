#!/usr/bin/env python3
"""
Environment Verification Script for NumPy/PyTorch Compatibility

This script checks if your Python environment has compatible versions of
NumPy and PyTorch installed, and tests basic functionality.

Usage:
    python verify_environment.py
"""

import sys


def check_imports():
    """Check if required packages can be imported."""
    print("=" * 70)
    print("CHECKING PACKAGE IMPORTS")
    print("=" * 70)

    packages = {
        'numpy': None,
        'torch': None,
        'torchvision': None,
        'cv2': None,
        'PIL': None,
    }

    for package_name in packages.keys():
        try:
            if package_name == 'cv2':
                import cv2
                packages[package_name] = cv2.__version__
                print(f"✓ OpenCV (cv2): {cv2.__version__}")
            elif package_name == 'PIL':
                import PIL
                packages[package_name] = PIL.__version__
                print(f"✓ Pillow (PIL): {PIL.__version__}")
            elif package_name == 'numpy':
                import numpy as np
                packages[package_name] = np.__version__
                print(f"✓ NumPy: {np.__version__}")
            elif package_name == 'torch':
                import torch
                packages[package_name] = torch.__version__
                print(f"✓ PyTorch: {torch.__version__}")
            elif package_name == 'torchvision':
                import torchvision
                packages[package_name] = torchvision.__version__
                print(f"✓ Torchvision: {torchvision.__version__}")
        except ImportError as e:
            print(f"✗ {package_name}: Not installed")
            packages[package_name] = None
        except Exception as e:
            print(f"✗ {package_name}: Error - {str(e)}")
            packages[package_name] = None

    return packages


def check_numpy_pytorch_compatibility(packages):
    """Check compatibility between NumPy and PyTorch versions."""
    print("\n" + "=" * 70)
    print("CHECKING NUMPY/PYTORCH COMPATIBILITY")
    print("=" * 70)

    if packages['numpy'] is None or packages['torch'] is None:
        print("✗ Cannot check compatibility: NumPy or PyTorch not installed")
        return False

    import numpy as np
    import torch

    numpy_version = packages['numpy']
    torch_version = packages['torch']

    print(f"NumPy version: {numpy_version}")
    print(f"PyTorch version: {torch_version}")

    # Check if NumPy is 2.x
    numpy_major = int(numpy_version.split('.')[0])

    # Check if PyTorch is older than 2.1
    torch_major = int(torch_version.split('.')[0])
    torch_minor = int(torch_version.split('.')[1])

    if numpy_major >= 2 and (torch_major < 2 or (torch_major == 2 and torch_minor < 1)):
        print("\n⚠️  WARNING: Potential compatibility issue detected!")
        print(f"   NumPy {numpy_version} may not be compatible with PyTorch {torch_version}")
        print("   Recommendation: Downgrade NumPy to <2.0 or upgrade PyTorch to >=2.1.0")
        print("\n   Quick fix: pip install 'numpy<2.0'")
        return False
    elif numpy_major >= 2 and torch_major >= 2 and torch_minor >= 1:
        print("\n⚠️  EXPERIMENTAL: NumPy 2.x with PyTorch 2.1+")
        print("   This combination has experimental support. May work but not fully tested.")
        return True
    else:
        print("\n✓ NumPy and PyTorch versions appear compatible")
        return True


def test_basic_functionality():
    """Test basic NumPy/PyTorch operations."""
    print("\n" + "=" * 70)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 70)

    try:
        import numpy as np
        import torch

        # Test 1: NumPy array creation
        print("\nTest 1: NumPy array creation...")
        np_array = np.array([1, 2, 3, 4, 5])
        print(f"✓ Created NumPy array: {np_array}")

        # Test 2: NumPy to PyTorch conversion
        print("\nTest 2: NumPy to PyTorch tensor conversion...")
        torch_tensor = torch.from_numpy(np_array)
        print(f"✓ Converted to PyTorch tensor: {torch_tensor}")

        # Test 3: PyTorch to NumPy conversion
        print("\nTest 3: PyTorch to NumPy array conversion...")
        back_to_numpy = torch_tensor.numpy()
        print(f"✓ Converted back to NumPy: {back_to_numpy}")

        # Test 4: CUDA availability
        print("\nTest 4: CUDA availability...")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
        else:
            print(f"✓ CUDA is not available (CPU mode)")

        # Test 5: Basic tensor operations
        print("\nTest 5: Basic tensor operations...")
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        c = a + b
        print(f"✓ Tensor addition: {a} + {b} = {c}")

        # Test 6: Neural network layer
        print("\nTest 6: Creating a simple neural network layer...")
        layer = torch.nn.Linear(10, 5)
        print(f"✓ Created Linear layer: {layer}")

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ TESTS FAILED: {str(e)}")
        print("\nThis error suggests an incompatibility issue.")
        print("Please refer to NUMPY_PYTORCH_COMPATIBILITY_FIX.md for solutions.")
        return False


def print_recommendations(packages, compatibility_ok, tests_passed):
    """Print recommendations based on test results."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if not compatibility_ok or not tests_passed:
        print("\n⚠️  Action Required:")
        print("\n1. Install compatible versions:")
        print("   pip install 'numpy<2.0' torch>=2.0.0")
        print("\n2. Or create a new environment:")
        print("   python -m venv ml_env")
        print("   source ml_env/bin/activate  # On Windows: ml_env\\Scripts\\activate")
        print("   pip install 'numpy<2.0' torch torchvision")
        print("\n3. For detailed solutions, see:")
        print("   NUMPY_PYTORCH_COMPATIBILITY_FIX.md")
    else:
        print("\n✓ Your environment looks good!")
        print("  You should be able to run most projects without issues.")
        print("\n  If you encounter problems, refer to:")
        print("  NUMPY_PYTORCH_COMPATIBILITY_FIX.md")

    print("\n" + "=" * 70)


def main():
    """Main function to run all checks."""
    print("\n" + "=" * 70)
    print("ENVIRONMENT VERIFICATION FOR AI/ML PROJECTS")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print("=" * 70)

    # Check imports
    packages = check_imports()

    # Check compatibility
    compatibility_ok = True
    if packages['numpy'] and packages['torch']:
        compatibility_ok = check_numpy_pytorch_compatibility(packages)

    # Test functionality
    tests_passed = False
    if packages['numpy'] and packages['torch']:
        tests_passed = test_basic_functionality()
    else:
        print("\n" + "=" * 70)
        print("SKIPPING FUNCTIONALITY TESTS")
        print("=" * 70)
        print("NumPy and/or PyTorch not installed")

    # Print recommendations
    print_recommendations(packages, compatibility_ok, tests_passed)

    # Exit with appropriate code
    if compatibility_ok and tests_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

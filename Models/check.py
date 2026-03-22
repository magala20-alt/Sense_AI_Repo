import sys
import os

print(f"Python Executable: {sys.executable}")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
except ModuleNotFoundError:
    print("❌ TensorFlow still not found.")
    print("\nYour Python is looking in these places:")
    for path in sys.path:
        print(f" - {path}")
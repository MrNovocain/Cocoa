
try:
    import cuml
    print("cuML is installed and importable.")
    print(f"cuML version: {cuml.__version__}")
except ImportError:
    print("cuML is NOT installed.")
except Exception as e:
    print(f"Error importing cuML: {e}")

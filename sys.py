import importlib

libraries = [
    "streamlit", "numpy", "opencv-python", "Pillow", "matplotlib",
    "scikit-image", "shap", "lime", "tensorflow", "scikit-learn",
    "deap", "h5py", "albumentations","numpy","cv2","os","pickle","PIL"
]

with open("requirements.txt", "w") as f:
    for lib in libraries:
        try:
            module = importlib.import_module(lib if lib != "opencv-python" else "cv2")
            version = getattr(module, "__version__", "unknown")
            f.write(f"{lib}=={version}\n")
        except ModuleNotFoundError:
            f.write(f"# {lib} not installed\n")

print("requirements.txt generated!")

import sys
import cv2
import numpy as np

print("Python:", sys.version)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

print("YuNet available:", hasattr(cv2, "FaceDetectorYN"))
print("SFace available:", hasattr(cv2, "FaceRecognizerSF"))

assert hasattr(cv2, "FaceDetectorYN")
assert hasattr(cv2, "FaceRecognizerSF")

print("\nEnvironment verification PASSED!")
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from fer import FER

img = cv2.imread("user.png")
predictions = DeepFace.analyze(img)
print(predictions['dominant_emotion'])
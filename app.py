import cv2
from tkinter import *
from PIL import Image, ImageTk

from deepface import DeepFace
from fer import FER
import face_capture as face

root = Tk()

root.title("Welcome to Music Recommendation")

root.minsize(width=400, height=400)
root.geometry("600x500")

def clicked():
    face.capture()    
    img = cv2.imread("user.png")
    predictions = DeepFace.analyze(img)
    print(predictions['dominant_emotion'])
    lbl.configure(text=predictions['dominant_emotion'])

lbl = Label(root, text="What is your mood")
lbl.grid()
btn = Button(root, text="Click me",
             fg="red", command=clicked)
btn.grid(column=1, row=0)

root.mainloop()
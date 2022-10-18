import tkinter
from tkinter import *
from PIL import Image, ImageTk
import face_capture
import emotion_recognition

root = Tk()

root.title("Welcome to Music Recommendation")

root.minsize(width=400, height=400)
root.geometry("600x500")

lbl = Label(root, text="What is your mood")
lbl.grid()


def clicked():
    face_capture.capture()
    emotion = emotion_recognition.predictions
    lbl.configure(text=emotion['dominant_emotion'])
    # image1 = Image.open("user.png")
    # test = ImageTk.PhotoImage(image1)
    # label1 = tkinter.Label(image=test)
    # label1.image = test
    # label1.place(x = 0 , y = 0)


btn = Button(root, text="Click me",
             fg="red", command=clicked)
btn.grid(column=1, row=0)

root.mainloop()

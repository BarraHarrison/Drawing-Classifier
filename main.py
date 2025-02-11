import pickle
import os.path

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog

import numpy as np 
import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class DrawingClassifier():

    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = None, None, None
        self.classifier = None
        self.project_name = None
        self.root = None
        self.image1 = None

        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 15

        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        msg = Tk()
        msg.withdraw()

        self.project_name = simpledialog.askstring("Project Name", "Please enter your project name down below.", parent=msg)
        if os.path.exists(self.project_name):
            with open(f"{self.project_name}/{self.project_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.class1 = data["c1"]
            self.class2 = data["c2"]
            self.class3 = data["c3"]
            self.class1_counter = data["c1c"]
            self.class2_counter = data["c2c"]
            self.class3_counter = data["c3c"]
            self.classifier = data["classifier"]
            self.project_name = data["project_name"]
        else:
            self.class1 = simpledialog.askstring("Class 1", "What is the first class called?", parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "What is the second class called?", parent=msg)
            self.class3 = simpledialog.askstring("Class 3", "What is the third class called?", parent=msg)

            self.class1_counter = 1
            self.class2_counter = 1
            self.class3_counter = 1

            self.classifier = LinearSVC()

            os.mkdir(self.project_name)
            os.chdir(self.project_name)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")

    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title(f"Python Drawing-Classifer v1.0 - {self.project_name}")

        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        button_frame = tkinter.Frame(self.root)
        button_frame.pack(fill=X, side=BOTTOM)

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        class1_button = Button(button_frame, text=self.class1, command=lambda: self.save(1))
        class1_button.grid(row=0, column=0, sticky=W + E)

        class2_button = Button(button_frame, text=self.class2, command=lambda: self.save(2))
        class2_button.grid(row=0, column=1, sticky=W + E)

        class3_button = Button(button_frame, text=self.class3, command=lambda: self.save(3))
        class3_button.grid(row=0, column=2, sticky=W + E)

        brush_minus_button = Button(button_frame, text="Brush-", command=self.brushminus)
        brush_minus_button.grid(row=1, column=0, sticky=W + E)

        clear_button = Button(button_frame, text="Clear", command=self.clear)
        clear_button.grid(row=1, column=1, sticky=W + E)

        brush_plus_button = Button(button_frame, text="Brush+", command=self.brushplus)
        brush_plus_button.grid(row=1, column=2, sticky=W + E)

        train_button = Button(button_frame, text="Train Model", command=self.train_model)
        train_button.grid(row=2, column=0, sticky=W + E)

        save_button = Button(button_frame, text="Save Model", command=self.save_model)
        save_button.grid(row=2, column=1, sticky=W + E)

        load_button = Button(button_frame, text="Load Model", command=self.load_model)
        load_button.grid(row=2, column=2, sticky=W + E)

        change_button = Button(button_frame, text="Change Model", command=self.rotate_model)
        change_button.grid(row=3, column=0, sticky=W + E)

        predict_button = Button(button_frame, text="Predict", command=self.predict)
        predict_button.grid(row=3, column=1, sticky=W + E)

        save_everything_button = Button(button_frame, text="Save Everything", command=self.save_everything)
        save_everything_button.grid(row=3, column=2, sticky=W + E)

        self.status_label = Label(button_frame, text=f"Current Model: {type(self.classifier).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

        
    def paint(self, event):
        pass

    def save(self, class_num):
        pass

    def brushminus(self):
        pass

    def brushplus(self):
        pass

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle(0, 0, 1000, 1000, fill="white")
        

    def train_model(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def rotate_model(self):
        pass

    def predict(self):
        pass

    def save_everything(self):
        pass
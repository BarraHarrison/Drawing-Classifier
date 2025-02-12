import pickle
import os

import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog

import numpy as np 
import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class DrawingClassifier():

    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None
        self.class1_counter, self.class2_counter, self.class3_counter = 1, 1, 1
        self.classifier = LinearSVC()
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
        else:
            self.class1 = simpledialog.askstring("Class 1", "What is the first class called?", parent=msg)
            self.class2 = simpledialog.askstring("Class 2", "What is the second class called?", parent=msg)
            self.class3 = simpledialog.askstring("Class 3", "What is the third class called?", parent=msg)

            os.mkdir(self.project_name)
            os.mkdir(f"{self.project_name}/{self.class1}")
            os.mkdir(f"{self.project_name}/{self.class2}")
            os.mkdir(f"{self.project_name}/{self.class3}")

    def init_gui(self):
        WIDTH, HEIGHT = 500

        self.root = Tk()
        self.root.title(f"Python Drawing-Classifer v1.0 - {self.project_name}")

        self.canvas = Canvas(self.root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), "white")
        self.draw = PIL.ImageDraw.Draw(self.image1)

        button_frame = Frame(self.root)
        button_frame.pack(fill=X, side=BOTTOM)

        buttons = [
            (self.class1, lambda: self.save(1)),
            (self.class2, lambda: self.save(2)),
            (self.class3, lambda: self.save(3)),
            ("Brush-", self.brushminus),
            ("Clear", self.clear),
            ("Brush+", self.brushplus),
            ("Train Model", self.train_model),
            ("Save Model", self.save_model),
            ("Load Model", self.load_model),
            ("Change Model", self.rotate_model),
            ("Predict", self.predict),
            ("Save Everything", self.save_everything)
        ]
        
        for i, (text, command) in enumerate(buttons):
            Button(button_frame, text=text, command=command).grid(row=i//3, column=i%3, sticky=W+E)

        self.status_label = Label(button_frame, text=f"Current Model: {type(self.classifier).__name__}")
        self.status_label.grid(row=4, column=1, sticky=W+E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

        
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    def save(self, class_num):
        self.image1.save("temporary.png")
        img = PIL.Image.open("temporary.png")
        img.thumbnail((50,50), PIL.Image.ANTIALIAS)

        if class_num == 1:
            img.save(f"{self.project_name}/{self.class1}/{self.class1_counter}.png", "PNG")
            self.class1_counter += 1
        elif class_num == 2:
            img.save(f"{self.project_name}/{self.class2}/{self.class2_counter}.png", "PNG")
            self.class2_counter += 1
        elif class_num == 3:
            img.save(f"{self.project_name}/{self.class3}/{self.class3_counter}.png", "PNG")
            self.class3_counter += 1

        self.clear()

    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def brushplus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle(0, 0, 1000, 1000, fill="white")


    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])

        for x in range(1, self.class1_counter):
            img = cv.imread(f"{self.project_name}/{self.class1}/{x}.png")[:, :, 0]
            img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)

        for x in range(1, self.class2_counter):
            img = cv.imread(f"{self.project_name}/{self.class2}/{x}.png")[:, :, 0]
            img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

        for x in range(1, self.class3_counter):
            img = cv.imread(f"{self.project_name}/{self.class3}/{x}.png")[:, :, 0]
            img.reshape(2500)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 3)

        img_list = img_list.reshape(self.class1_counter - 1 + self.class2_counter - 1 + self.class3_counter - 1, 2500)

        self.classifier.fit(img_list, class_list)
        tkinter.messagebox.showinfo("Python Drawing Classifier", "Model Successfully Trained", parent=self.root)

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.classifier, f)
        tkinter.messagebox.showinfo("Python Drawing Classifier", "Model Successfully saved!", parent=self.root)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.classifier = pickle.load(f)
            tkinter.messagebox.showinfo("Python Drawing Classifier", "Model Successfully saved!", parent=self.root)

    def rotate_model(self):
        if isinstance(self.classifier, LinearSVC):
            self.classifier = KNeighborsClassifier()
        elif isinstance(self.classifier, KNeighborsClassifier):
            self.classifier = LogisticRegression()
        elif isinstance(self.classifier, LogisticRegression):
            self.classifier = DecisionTreeClassifier()
        elif isinstance(self.classifier, DecisionTreeClassifier):
            self.classifier = RandomForestClassifier()
        elif isinstance(self.classifier, RandomForestClassifier):
            self.classifier = GaussianNB()
        elif isinstance(self.classifier, GaussianNB):
            self.classifier = LinearSVC()

        self.status_label.config(text=f"Current Model: {type(self.classifier).__name__}")

    def predict(self):
        self.image1.save("temporary.png")
        img = PIL.Image.open("temporary.png")
        img.thumbnail((50,50), PIL.Image.ANTIALIAS)
        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.classifier.predict([img])
        if prediction[0] == 1:
            tkinter.messagebox.showinfo("Python Drawing Classifier", f"The drawing is most likely a {self.class1}", parent=self.root)
        elif prediction[0] == 2:
            tkinter.messagebox.showinfo("Python Drawing Classifier", f"The drawing is most likely a {self.class2}", parent=self.root)
        elif prediction[0] == 3:
            tkinter.messagebox.showinfo("Python Drawing Classifier", f"The drawing is most likely a {self.class3}", parent=self.root)

    def save_everything(self):
        data = {"c1": self.class1,
                "c2": self.class2,
                "c3": self.class3,
                "c1c": self.class1_counter,
                "c2c": self.class2_counter,
                "c3c": self.class3_counter,
                "classifier": self.classifier,
                "project_name": self.project_name}
        
        with open(f"{self.project_name}/{self.project_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("Python Drawing Classifier", "Project Successfully saved!", parent=self.root)

    
    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?", "Do you want to save your work?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()


DrawingClassifier()
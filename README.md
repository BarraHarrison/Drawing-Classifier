# Drawing Classifier 🖌️

## Introduction
This project uses Machine Learning to classify drawings into three classes. It includes functionality to:
1. Create a new classification project or load an existing one.
2. Collect and save drawing samples for each class.
3. Train a machine learning model to classify new drawings.
4. Save and load the trained model.
5. Save all project data for future use.

Below are the key sections of the code explained.

---

## 1. Imports and Library Setup 📚
This section imports the necessary libraries for the project:
- **pickle** for saving and loading Python objects (like model data).
- **os.path** for handling file paths and checking if certain files or directories exist.
- **tkinter** for building the graphical user interface (GUI). I also use dialog modules (`simpledialog`, `filedialog`) and `messagebox` for user interactions (e.g., prompts, saving dialogs).
- **PIL (Pillow)** for image manipulation (e.g., resizing, drawing).
- **cv2** (OpenCV) and **numpy** for reading, processing, and reshaping images into arrays.
- **scikit-learn** classifiers such as `LinearSVC`, `GaussianNB`, `DecisionTreeClassifier`, etc., for training the Machine Learning model.

---

## 2. Classes Prompt Dialog 🗂️
**Function: `classes_prompt(self)`**

- Prompts the user to enter or load an existing project name.
- If the project exists, it loads the saved data (classes, counters, classifier, etc.) from a `.pickle` file.
- If the project does not exist, it creates a new one by:
  - Asking the user to name three different classes (e.g., "circle", “square”, “triangle”).
  - Initializing counters for each class to track how many samples are saved.
  - Defaulting the classifier to `LinearSVC`.
  - Creating folders for each class and storing them in a new project directory.

This approach allows for quickly resuming an existing classification project or starting a fresh one with three distinct classes.

---

## 3. Saving Drawings ✏️
**Function: `save(self, class_num)`**

- Saves the user’s current drawing from the Tkinter canvas to a temporary file, then:
  - Opens and resizes it to a 50×50 thumbnail (keeping it small for efficient training).
  - Depending on which class number is passed in (`1`, `2`, or `3`), the drawing is saved to the corresponding class folder.
  - Increments the counter for that class, ensuring the next saved file has a unique name.
- Finally, it clears the drawing area to allow the user to draw and save the next sample.

---

## 4. Training the Model 🤖
**Function: `train_model(self)`**

- Loads all saved images from each class folder and converts them into grayscale arrays.
- Reshapes each image into a 1D array of 2,500 pixels (since 50×50 = 2,500).
- Appends the reshaped arrays to `img_list` and the corresponding class labels (`1`, `2`, or `3`) to `class_list`.
- Trains the chosen classifier (`self.classifier`) with the `fit` method using the prepared arrays and labels.
- Displays a message box confirming successful training.

This step is critical, as it converts the user’s drawings into numeric data the classifier can understand, then teaches it how to distinguish among the three classes.

---

## 5. Saving the Model 💾
**Function: `save_model(self)`**

- Prompts the user for a file path (using a “Save As” dialog).
- Serializes (pickles) the trained classifier into a `.pickle` file.
- Shows a confirmation message that the model was saved.

Saving the model in this way allows you to reuse or share the trained classifier without having to retrain from scratch.

---

## 6. Loading the Model 📂
**Function: `load_model(self)`**

- Prompts the user to select a `.pickle` file that contains a previously saved classifier.
- Loads the model from the selected file into `self.classifier`.
- Displays a confirmation message that the model was loaded.

This feature is useful when you already have a trained model and want to classify new drawings without going through the training process again.

---

## 7. Saving All Project Data 🌐
**Function: `save_everything(self)`**

- Creates a dictionary that holds all vital project information, including:
  - Class names (`c1`, `c2`, `c3`)
  - Class counters (`c1c`, `c2c`, `c3c`)
  - The trained classifier object
  - The project name
- Stores this dictionary in a `.pickle` file inside the project folder.
- Confirms to the user that the project is successfully saved.

This ensures not only the model but also all counters and class names are stored for complete project continuity.

---

## 8. Application Closing Workflow 🚪
**Function: `on_closing(self)`**

- Before exiting, it asks the user whether they want to save their work.
- If yes, it calls `save_everything()` to store all project data safely.
- Then, it closes the main Tkinter window and ends the program.

This prompts the user one last time to preserve their data, preventing accidental loss.

---

## Problems Faced 😅
- **Tkinter Background Color**: Changing the default Tkinter background to white proved difficult due to deprecated methods and inconsistent behavior across versions.
- **GUI Creation Issues**: In some ML projects, handling different Tkinter versions can cause compatibility problems, leading to additional effort in ensuring the interface works correctly on various setups.

---

## Conclusion
This Drawing Classifier project showcases how you can use a straightforward Tkinter GUI alongside various Python libraries to create, train, and save machine learning classification models for drawings. By organizing the workflow into distinct functions, you can easily collect data, train your model, and manage your entire project’s metadata. 

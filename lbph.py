# lbph supervised learning model using harrcascade


import cv2
import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

# Paths and configurations
DATASET_DIR = "dataset"
EXCEL_FILE = "candidate_data.xlsx"
MODEL_FILE = "face_recognition_model.xml"
NUM_PHOTOS_PER_CANDIDATE = 20
INTERVAL_SECONDS = 0.25

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Excel setup: Create the Excel file if it doesn't exist
if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=["Name", "Age", "Class", "Gender", "Photo Path"])
    df.to_excel(EXCEL_FILE, index=False)

# Function to capture images and detect faces
def capture_images(name, age, _class, gender):
    candidate_dir = os.path.join(DATASET_DIR, name)
    if not os.path.exists(candidate_dir):
        os.makedirs(candidate_dir)

    cap = cv2.VideoCapture(0)
    count = 0
    photo_paths = []

    while count < NUM_PHOTOS_PER_CANDIDATE:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            photo_path = f"{candidate_dir}/{name}_{count}.jpg"
            cv2.imwrite(photo_path, face_img)
            photo_paths.append(photo_path)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow("Face Capture", frame)

        time.sleep(INTERVAL_SECONDS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Append the data to Excel using pd.concat
    df = pd.read_excel(EXCEL_FILE)
    new_rows = pd.DataFrame([{"Name": name, "Age": age, "Class": _class, "Gender": gender, "Photo Path": path} for path in photo_paths])
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)

# Function to collect face data for training
def collect_training_data():
    X = []
    y = []
    df = pd.read_excel(EXCEL_FILE)

    for _, row in df.iterrows():
        img = cv2.imread(row["Photo Path"], cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y_coord, w, h) in faces:
            face_img = img[y_coord:y_coord+h, x:x+w]
            X.append(face_img)
            y.append(row["Name"])  # Append the label to the list
    
    return X, y

# Train the LBPH model and save it to disk
def train_lbph_model(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Initialize LBPH Face Recognizer
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X, y_encoded)

    # Save the model to disk
    model.write(MODEL_FILE)

    return model, label_encoder

# Function to load the trained model
def load_model():
    if os.path.exists(MODEL_FILE):
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(MODEL_FILE)
        return model
    return None

# Function to predict the face
def predict_face(model, label_encoder, img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if len(faces) == 0:
        return None, None  # Return None for both label and confidence if no face is detected

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]

        # Ensure the image is in the correct format for prediction
        try:
            label, confidence = model.predict(face_img)
            predicted_label = label_encoder.inverse_transform([label])[0]
            return predicted_label, confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None  # Return None if there's an error during prediction

    return None, None  # Default return in case something unexpected happens


# Live face recognition
def live_recognition():
    model = load_model()
    if model is None:
        messagebox.showwarning("Model Error", "No trained model found. Please train the model first.")
        return

    label_encoder = LabelEncoder()
    df = pd.read_excel(EXCEL_FILE)
    label_encoder.fit(df["Name"])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            prediction, confidence = predict_face(model, label_encoder, face_img)

            # Display prediction on frame
            if prediction is not None:
                cv2.putText(frame, f"{prediction} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Functionality
def start_capture():
    name = entry_name.get()
    age = entry_age.get()
    _class = entry_class.get()
    gender = entry_gender.get()

    if name and age and _class and gender:
        capture_images(name, age, _class, gender)
        messagebox.showinfo("Success", f"Images captured for {name}")
    else:
        messagebox.showwarning("Input Error", "Please provide all inputs (Name, Age, Class, Gender)")

def train_model():
    X, y = collect_training_data()
    if X and y:
        model, label_encoder = train_lbph_model(X, y)
        messagebox.showinfo("Success", "Model trained successfully")
    else:
        messagebox.showwarning("Data Error", "No training data found")

# GUI Setup
root = Tk()
root.title("Face Recognition System")

Label(root, text="Name").grid(row=0, column=0)
entry_name = Entry(root)
entry_name.grid(row=0, column=1)

Label(root, text="Age").grid(row=1, column=0)
entry_age = Entry(root)
entry_age.grid(row=1, column=1)

Label(root, text="Class").grid(row=2, column=0)
entry_class = Entry(root)
entry_class.grid(row=2, column=1)

Label(root, text="Gender").grid(row=3, column=0)
entry_gender = Entry(root)
entry_gender.grid(row=3, column=1)

Button(root, text="Capture Images", command=start_capture).grid(row=4, column=0, pady=10)
Button(root, text="Train Model", command=train_model).grid(row=4, column=1, pady=10)
Button(root, text="Live Recognition", command=live_recognition).grid(row=5, column=0, pady=10, columnspan=2)

root.mainloop()

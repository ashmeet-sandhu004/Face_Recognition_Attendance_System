import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import face_recognition
from datetime import datetime

# Create necessary directories
if not os.path.exists("TrainingImage"):
    os.makedirs("TrainingImage")
if not os.path.exists("TrainingImageLabel"):
    os.makedirs("TrainingImageLabel")
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# Functions (same as original, no change needed)

def clear_text(entry):
    entry.delete(0, tk.END)

def take_images():
    Id = txt.get()
    name = txt2.get()

    if Id.isdigit() and name.isalpha():
        cam = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, img = cam.read()
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for face recognition
            face_locations = face_recognition.face_locations(rgb_img)

            for (top, right, bottom, left) in face_locations:
                count += 1
                face_image = rgb_img[top:bottom, left:right]
                face_image_pil = Image.fromarray(face_image)
                face_image_pil.save(f"TrainingImage/{name}.{Id}.{count}.jpg")

                cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

            cv2.imshow("Taking Images", img)
            if cv2.waitKey(1) == 27 or count >= 100:  # Escape key or 100 samples
                break

        cam.release()
        cv2.destroyAllWindows()

        res = str(f"Images Saved for ID: {Id}, Name: {name}")
        message.configure(text=res)

        row = [Id, name]
        with open("StudentDetails.csv", "a") as csvFile:
            writer = pd.DataFrame([row])
            writer.to_csv(csvFile, header=False, index=False)
    else:
        messagebox.showerror("Input Error", "Enter a valid ID (numeric) and Name (alphabetic).")

def train_images():
    known_face_encodings = []
    known_face_names = []

    # Load each image and extract face encoding
    for root, dirs, files in os.walk("TrainingImage"):
        for file in files:
            if file.endswith("jpg"):
                path = os.path.join(root, file)
                img = Image.open(path)
                img_np = np.array(img)
                rgb_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_img)
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

                if len(face_encodings) > 0:
                    known_face_encodings.append(face_encodings[0])  # We assume one face per image
                    Id = int(file.split(".")[1])
                    known_face_names.append(Id)

    # Save the trained data
    np.save("TrainingImageLabel/known_face_encodings.npy", known_face_encodings)
    np.save("TrainingImageLabel/known_face_names.npy", known_face_names)

    messagebox.showinfo("Success", "Images Trained Successfully!")

def track_images():
    # Check if training files exist
    if not os.path.exists("TrainingImageLabel/known_face_encodings.npy") or not os.path.exists("TrainingImageLabel/known_face_names.npy"):
        messagebox.showerror("Error", "Training data not found. Please train the images first.")
        return

    known_face_encodings = np.load("TrainingImageLabel/known_face_encodings.npy", allow_pickle=True)
    known_face_names = np.load("TrainingImageLabel/known_face_names.npy", allow_pickle=True)

    cam = cv2.VideoCapture(0)
    recognized_faces = []  # List to track faces that have already been recognized

    while True:
        ret, img = cam.read()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            if name != "Unknown" and name not in recognized_faces:
                # Mark attendance and add to the list of recognized faces
                take_attendance(name)
                recognized_faces.append(name)

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Tracking Images", img)
        if cv2.waitKey(1) == 27:  # Escape key
            break

    cam.release()
    cv2.destroyAllWindows()


def take_attendance(name):
    # Record the attendance with timestamp
    with open("Attendance/attendance.csv", "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{name}, {timestamp}\n")
    print(f"Attendance marked for: {name}")

# GUI Setup
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("800x500")
window.configure(bg="#f1f1f1")

# Title Label
title_label = tk.Label(window, text="Face Recognition Attendance System", bg="#f1f1f1", fg="#2c3e50", font=("Helvetica", 18, "bold"))
title_label.pack(pady=20)

# Labels and Entries
frame = tk.Frame(window, bg="#f1f1f1")
frame.pack(pady=20)

lbl = tk.Label(frame, text="Enter ID:", bg="#f1f1f1", fg="#2c3e50", font=("Helvetica", 12))
lbl.grid(row=0, column=0, padx=10, pady=10)
txt = tk.Entry(frame, width=20, font=("Helvetica", 12))
txt.grid(row=0, column=1, padx=10, pady=10)

lbl2 = tk.Label(frame, text="Enter Name:", bg="#f1f1f1", fg="#2c3e50", font=("Helvetica", 12))
lbl2.grid(row=1, column=0, padx=10, pady=10)
txt2 = tk.Entry(frame, width=20, font=("Helvetica", 12))
txt2.grid(row=1, column=1, padx=10, pady=10)

# Buttons
button_frame = tk.Frame(window, bg="#f1f1f1")
button_frame.pack(pady=30)

takeImg = tk.Button(button_frame, text="Take Images", command=take_images, bg="#3498db", fg="white", font=("Helvetica", 14, "bold"), relief="solid", width=20)
takeImg.grid(row=0, column=0, padx=20, pady=10)

trainImg = tk.Button(button_frame, text="Train Images", command=train_images, bg="#2ecc71", fg="white", font=("Helvetica", 14, "bold"), relief="solid", width=20)
trainImg.grid(row=0, column=1, padx=20, pady=10)

trackImg = tk.Button(button_frame, text="Track Images", command=track_images, bg="#e74c3c", fg="white", font=("Helvetica", 14, "bold"), relief="solid", width=20)
trackImg.grid(row=1, column=0, padx=20, pady=10)

takeAttendance = tk.Button(button_frame, text="Take Attendance", command=lambda: track_images(), bg="#0984e3", fg="white", font=("Helvetica", 14, "bold"), relief="solid", width=20)
takeAttendance.grid(row=1, column=1, padx=20, pady=10)

# Status Message
message = tk.Label(window, text="", bg="#f1f1f1", fg="#2c3e50", font=("Helvetica", 12))
message.pack(pady=20)

# Run the GUI
window.mainloop()

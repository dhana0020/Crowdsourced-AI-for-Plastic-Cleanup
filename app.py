from flask import Flask, render_template, request, redirect
import os
import csv
import cv2
from datetime import datetime
import pandas as pd
import requests
from ultralytics import YOLO

app = Flask(__name__)

# === Download model from Google Drive ===
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1lLcJbJa86pl-cV8PrETPoa19bd-fXOQw"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model from Google Drive...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model download complete.")

# Load YOLO model
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CSV_FILE = "uploads.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename_original", "filename_result", "detections", "upload_date", "upload_time", "location"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Run YOLO detection
        image = cv2.imread(filepath)
        results = model(image)[0]

        detections = []
        for box in results.boxes:
            cls = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append(f"{cls}:{conf:.2f}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(image, f"{cls} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Save result image
        result_filename = f"result_{filename}"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, image)

        # Date/time
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Location from frontend
        lat = request.form.get("latitude", "")
        lon = request.form.get("longitude", "")
        location_str = f"{lat}, {lon}" if lat and lon else "Unknown"

        # Save to CSV
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, result_filename, ", ".join(detections), date_str, time_str, location_str])

        return render_template("index.html",
                               result_image=result_filename,
                               detections=", ".join(detections),
                               date=date_str,
                               time=time_str,
                               location=location_str)

@app.route("/map")
def map_page():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
            df = df.fillna("")
            df = df.astype(str)
            data = df.to_dict(orient="records")
        except Exception as e:
            print("CSV read error:", e)
            data = []
    else:
        data = []
    return render_template("map.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)

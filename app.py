from flask import Flask, render_template, request, redirect
from ultralytics import YOLO
import cv2, os, csv
from datetime import datetime
import pandas as pd
import gdown  # <-- Added for Google Drive download

# ------------------ CONFIG ------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
GOOGLE_DRIVE_ID = "1lLcJbJa86pl-cV8PrETPoa19bd-fXOQw"  # Your Google Drive file ID

UPLOAD_FOLDER = "static/uploads"
CSV_FILE = "uploads.csv"

# Create necessary folders
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ DOWNLOAD MODEL IF MISSING ------------------
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)

# Load YOLO model
model = YOLO(MODEL_PATH)

# ------------------ INIT CSV ------------------
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename_original", "filename_result", "detections",
                         "upload_date", "upload_time", "location"])

# ------------------ FLASK APP ------------------
app = Flask(__name__)

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
            writer.writerow([filename, result_filename, ", ".join(detections),
                             date_str, time_str, location_str])

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

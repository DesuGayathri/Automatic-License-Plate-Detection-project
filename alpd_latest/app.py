import cv2
import os
import easyocr
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from ultralytics import YOLO

# Flask and DB setup
app = Flask(__name__)
CORS(app)
Base = declarative_base()
engine = create_engine('sqlite:///plates.db')
Session = sessionmaker(bind=engine)
session = Session()

# YOLO + OCR setup
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])
cap = cv2.VideoCapture(0)
#cv2.VideoCapture("http://100.78.63.191:8080/video")

# DB Table
class PlateLog(Base):
    __tablename__ = 'plate_logs'
    id = Column(Integer, primary_key=True)
    plate_number = Column(String)
    in_time = Column(DateTime)
    out_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Float, default=0.0)
    duration_hours = Column(Float, default=0.0)
    duration_days = Column(Float, default=0.0)

Base.metadata.create_all(engine)

detected_once = {}

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            text = reader.readtext(roi)
            if text:
                plate_number = text[0][1].upper().replace(" ", "")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, plate_number, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                now = datetime.now()
                log = session.query(PlateLog).filter_by(plate_number=plate_number).order_by(PlateLog.in_time.desc()).first()

                if log and log.out_time is None:
                    log.out_time = now
                    duration = now - log.in_time
                    log.duration_hours = round(duration.total_seconds() / 3600, 2)
                    log.duration_days = round(duration.total_seconds() / (3600*24), 2)
                    log.duration_minutes = round(duration.total_seconds() / 60, 2)
                    session.commit()
                elif not log or (log.out_time is not None and (now - log.out_time).seconds > 10):
                    new_log = PlateLog(plate_number=plate_number, in_time=now)
                    session.add(new_log)
                    session.commit()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/details')
def details():
    logs = session.query(PlateLog).order_by(PlateLog.in_time.desc()).all()
    return render_template("details.html", logs=logs)

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host="127.0.0.1", port=5000)


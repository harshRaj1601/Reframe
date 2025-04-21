import os
import cv2
import shutil
import sqlite3
import threading
from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
from ultralytics import YOLO
import uuid
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Configuration
app.config.update(
    UPLOAD_FOLDER='uploads',
    OUTPUT_FOLDER='output_clips',
    TEMP_FOLDER='temp',
    MAX_CONTENT_LENGTH=1024 * 1024 * 1024,  # 1GB
    ALLOWED_EXTENSIONS={'mp4', 'avi', 'mov', 'mkv'},
    DATABASE='video_history.db',
    YOLO_MODEL='yolov8n.pt'
)

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['TEMP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

app.logger.setLevel(logging.DEBUG)

def init_db():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS video_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT,
                processed_filename TEXT,
                upload_date DATETIME,
                status TEXT,
                job_id TEXT,
                start_time REAL,
                end_time REAL,
                target_object TEXT
            )
        ''')

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video(video_path, output_dir, job_id, start_time, end_time, target_object, model):
    """Process video to create shorts focusing on the target object"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Set video parameters
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_height = 1280
    target_width = 720
    
    # Prepare output video
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{job_id}_final.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if current_time > end_time:
                break

            # Detect objects
            results = model(frame)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            # Find target object
            target_boxes = []
            for box, cls in zip(boxes, classes):
                if results.names[int(cls)] == target_object:
                    target_boxes.append(box)

            if target_boxes:
                # Use the first detected target object
                x1, y1, x2, y2 = map(int, target_boxes[0])
                
                # Calculate center of the object
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Calculate crop region
                crop_x = max(0, min(center_x - target_width // 2, frame_width - target_width))
                crop_y = max(0, min(center_y - target_height // 2, frame_height - target_height))
                
                # Crop and resize
                cropped = frame[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
                if cropped.shape[:2] != (target_height, target_width):
                    cropped = cv2.resize(cropped, (target_width, target_height))
                
                out.write(cropped)

    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        return None
    finally:
        cap.release()
        out.release()

    return output_path

def process_video_background(video_path, job_id, start_time, end_time, target_object):
    """Process video in background and update database"""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    app.logger.debug(f"Starting video processing for job_id: {job_id}")
    
    try:
        model = YOLO(app.config['YOLO_MODEL'])
        final_path = process_video(video_path, output_dir, job_id, start_time, end_time, target_object, model)
        
        with sqlite3.connect(app.config['DATABASE']) as conn:
            if final_path and os.path.exists(final_path):
                conn.execute(
                    'UPDATE video_history SET status = ?, processed_filename = ? WHERE job_id = ?',
                    ('completed', final_path, job_id)
                )
            else:
                conn.execute(
                    'UPDATE video_history SET status = ? WHERE job_id = ?',
                    ('error', job_id)
                )
    except Exception as e:
        app.logger.error(f"Error in background processing: {str(e)}")
        with sqlite3.connect(app.config['DATABASE']) as conn:
            conn.execute(
                'UPDATE video_history SET status = ? WHERE job_id = ?',
                ('error', job_id)
            )

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process/<temp_id>')
def process_page(temp_id):
    temp_files = [f for f in os.listdir(app.config['TEMP_FOLDER']) if f.startswith(temp_id)]
    if not temp_files:
        return "File not found", 404
    return render_template('process.html')

@app.route('/upload-initial', methods=['POST'])
def upload_initial():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['video']
        if not file or not file.filename:
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        temp_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['TEMP_FOLDER'], f"{temp_id}_{filename}"))
        
        return jsonify({'temp_id': temp_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process-video', methods=['POST'])
def process_video_route():
    try:
        temp_id = request.form.get('temp_id')
        temp_files = [f for f in os.listdir(app.config['TEMP_FOLDER']) if f.startswith(temp_id)]
        if not temp_files:
            return jsonify({'error': 'Temporary file not found'}), 404

        job_id = str(uuid.uuid4())
        temp_path = os.path.join(app.config['TEMP_FOLDER'], temp_files[0])
        final_filename = temp_files[0].replace(temp_id, job_id)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        
        shutil.move(temp_path, upload_path)

        with sqlite3.connect(app.config['DATABASE']) as conn:
            conn.execute(
                'INSERT INTO video_history (original_filename, upload_date, status, job_id, start_time, end_time, target_object) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (final_filename, datetime.now(), 'processing', job_id, 
                 float(request.form.get('startTime', 0)),
                 float(request.form.get('endTime', 0)),
                 request.form.get('targetObject', 'person'))
            )

        thread = threading.Thread(
            target=process_video_background,
            args=(upload_path, job_id, 
                  float(request.form.get('startTime', 0)),
                  float(request.form.get('endTime', 0)),
                  request.form.get('targetObject', 'person'))
        )
        thread.start()

        return jsonify({'job_id': job_id, 'status': 'processing'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/temp-video/<temp_id>')
def temp_video(temp_id):
    try:
        temp_files = [f for f in os.listdir(app.config['TEMP_FOLDER']) if f.startswith(temp_id)]
        if not temp_files:
            return "File not found", 404
        return send_file(os.path.join(app.config['TEMP_FOLDER'], temp_files[0]))
    except Exception as e:
        return str(e), 500

@app.route('/status/<job_id>')
def check_status(job_id):
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            cursor = conn.execute('SELECT status, processed_filename FROM video_history WHERE job_id = ?', (job_id,))
            result = cursor.fetchone()
            
        if not result:
            return jsonify({'error': 'Job not found'}), 404
            
        status, processed_filename = result
        return jsonify({
            'status': status,
            'output_file': processed_filename if status == 'completed' else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<job_id>')
def download_output(job_id):
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            cursor = conn.execute('SELECT processed_filename FROM video_history WHERE job_id = ?', (job_id,))
            result = cursor.fetchone()
            
        if not result or not result[0] or not os.path.exists(result[0]):
            return jsonify({'error': 'Output not found'}), 404
            
        return send_file(result[0], as_attachment=True, download_name="final_output.mp4")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
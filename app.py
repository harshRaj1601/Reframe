import os
import sys
import cv2
import shutil
import sqlite3
import threading
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, after_this_request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
from ultralytics import YOLO
import uuid
import json
import numpy as np



import logging

import zipfile

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_clips'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['DATABASE'] = 'video_history.db'
app.config['YOLO_MODEL'] = 'yolov8n.pt'  # Path to the YOLO model

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

app.logger.setLevel(logging.DEBUG) #Add to the top

# Database setup
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''
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
    conn.commit()
    conn.close()

init_db()

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def reframe_video_to_shorts_in_clips(video_path, output_dir, job_id, start_time, end_time, target_object="person", model_path="yolov8n.pt", shorts_size=(720, 1280),
                                    min_object_presence=0.5, transition_frames=5, final_output_path="final_output.mp4"):
    """Reframe video to focus on a particular object, creating intelligent shorts with smooth transitions.  Combines clips into one final video."""
    app.logger.debug(f"Starting reframe_video_to_shorts_in_clips for job_id: {job_id}")#Add

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None # Return None to indicate failure

    # Set starting position for processing
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # Set in milliseconds

    try:
        model = YOLO(model_path)
        app.logger.debug(f"YOLO model loaded successfully for job_id: {job_id}")#Add

    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        app.logger.error(f"Error loading YOLO model: {e} for job_id: {job_id}")#Add

        cap.release()
        return None  # Return None to indicate failure

    shorts_width, shorts_height = shorts_size
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Not needed anymore
    max_clip_length = int(fps * 30)  # Max 30 seconds clip

    os.makedirs(output_dir, exist_ok=True)

    current_clip_index = 1
    frame_count = 0
    current_clip_frames = []
    #object_class_id = 0 #changed to not use int and class id

    # Fetch class names from the model
    class_names = model.names  # Access class names directly from the model

    def calculate_cropping_offsets(frame, person_boxes, last_offsets, class_names, target_object):  # Separate function for calculations
        """Calculate the x and y offsets based on object detection. Returns (x_offset, y_offset)"""
        shorts_width, shorts_height = shorts_size
        boxes_of_interest = [box for i, box in enumerate(person_boxes) if class_names[int(person_boxes[i][5])] == target_object]

        if len(boxes_of_interest) > 0:
            x1, y1, x2, y2, confidence, class_id = boxes_of_interest[0] #get the first box
            x1, y1, x2, y2 = map(int, [x1,y1,x2,y2])  # Convert coordinates to integers

            object_center_x = (x1 + x2) // 2
            object_center_y = (y1 + y2) // 2

            x_offset = object_center_x - shorts_width // 2
            y_offset = object_center_y - shorts_height // 2

            # Adjust offsets to stay within frame bounds
            x_offset = max(0, min(x_offset, frame.shape[1] - shorts_width))
            y_offset = max(0, min(y_offset, frame.shape[0] - shorts_height))
            last_offsets = (x_offset, y_offset)  # Store for next frame

        else:
            x_offset, y_offset = last_offsets  # Keep previous position

        return x_offset, y_offset, last_offsets

    def create_clip(frames, index):
       """Creates video clip, returns path or None"""
       if not frames:
           return None

       output_path = os.path.join(output_dir, f"{job_id}_shorts_clip_{index:02d}.mp4")
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       output_writer = cv2.VideoWriter(output_path, fourcc, fps, shorts_size)

       for frame in frames:
           output_writer.write(frame)  # write the frame

       output_writer.release()
       print(f"Clip saved at: {output_path}")
       return output_path

    def apply_transition(frames1, frames2, blend_frames):
        """Applies smooth transition between two sets of frames"""
        transitioned_frames = []  # init list
        for i in range(blend_frames):  # range is the transition frame so start and end of the clip can have a blend
            alpha = i / blend_frames  # calculate transparency

            # check if frames1 list is not empty and frames2 list is also not empty
            if frames1 and frames2:
                blended_frame = cv2.addWeighted(frames1[-blend_frames + i], 1 - alpha, frames2[i], alpha, 0)
                transitioned_frames.append(blended_frame)  # If it passes, blend both lists of frames
            elif frames1:
                transitioned_frames.append(frames1[-blend_frames + i])  # Transition is the first clip is bigger than second
            elif frames2:
                transitioned_frames.append(frames2[i])  # Vice versa.

        return transitioned_frames

    # Initialize last_good_offsets *before* the main loop to ensure it's always defined
    last_good_offsets = (0, 0)
    clip_paths = [] # Store paths to created clips
    processing = True #Variable to check if processing is valid

    while True:
        ret, frame = cap.read()

        # Break based on specified END TIME or If read fails
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Get current position in seconds

        if not ret or current_time > end_time: #end if we reach end time or are invalid
            processing = False #If we broke, set flag to false
            break #break

        results = model.predict(frame, show=False)
        detections = results[0].boxes
        clss = detections.cls.cpu().numpy()
        boxes = detections.xyxy.cpu().numpy()
        conf = detections.conf.cpu().numpy() # Access confidence scores

        #Combine detections boxes and conf
        person_boxes = np.concatenate((boxes, conf[:, None], clss[:, None]), axis=1)

        x_offset, y_offset, last_good_offsets = calculate_cropping_offsets(frame, person_boxes, last_good_offsets, class_names, target_object)

        # Crop the frame
        cropped_frame = frame[y_offset:y_offset + shorts_height, x_offset:x_offset + shorts_width]
        resized_frame = cv2.resize(cropped_frame, (shorts_width, shorts_height))

        # Object presence check (simplified: is the object detected)
        boxes_of_interest = [box for i, box in enumerate(person_boxes) if class_names[int(person_boxes[i][5])] == target_object] #check to make sure the boxes of interest exist

        if len(boxes_of_interest) > 0: # changed from distance calculation
            current_clip_frames.append(resized_frame)  # Append the resized frame
            frame_count += 1

        # Clip management
        if (len(boxes_of_interest) == 0 or frame_count >= max_clip_length or not ret or current_time > end_time):  # End clip
            if len(current_clip_frames) > 0:
                # SMOOTHING:  Apply transition at the END of the clip
                if current_clip_frames and len(current_clip_frames) > transition_frames:
                    transition_out_frames = apply_transition(current_clip_frames, [], transition_frames)
                    # Remove the transition from the old frames
                    del current_clip_frames[-transition_frames:]

                    # Extend the end to the new frames from the transition
                    current_clip_frames.extend(transition_out_frames)

                output_path = create_clip(current_clip_frames, current_clip_index)
                if output_path:  # Check if the clip was created
                    clip_paths.append(output_path)
                    current_clip_index += 1
                    frame_count = 0
                    current_clip_frames = []
                last_good_offsets = (0, 0)  # Reset the last position

        # Display preview frame - now using resized_frame
        if len(current_clip_frames) > 0:  # Check if there are frames to preview
            preview_frame = current_clip_frames[-1].copy()  # Last frame
            #The code was refactored so we must rebuild this list to find it's bounding box
            x_offset, y_offset, last_good_offsets = calculate_cropping_offsets(frame, person_boxes, last_good_offsets, class_names, target_object) #Calculate the offset
            boxes_of_interest = [box for i, box in enumerate(person_boxes) if class_names[int(person_boxes[i][5])] == target_object] #Rebuild list
            for box in boxes_of_interest:  # Draw detected object on preview
                x1, y1, x2, y2, confidence, class_id = box  # Extract additional info
                x1, y1, x2, y2 = map(int, [x1,y1,x2,y2]) #map and convert to int
                x1 = max(0, x1 - x_offset)  # Offset for draw rectangle on output video
                x2 = max(0, x2 - x_offset)  # Offset for draw rectangle on output video
                y1 = max(0, y1 - y_offset)  # Offset for draw rectangle on output video
                y2 = max(0, y2 - y_offset)  # Offset for draw rectangle on output video
                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(preview_frame, str(class_names[int(class_id)]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # cv2.imshow('Shorts Frame (Preview)', preview_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"All clips saved in {output_dir}")
    final_output_path = os.path.join(output_dir, f"{job_id}_final_video.mp4") #Set final path
    combine_clips(clip_paths, final_output_path, shorts_size, fps) 
    print(f"Combined video saved at: {final_output_path}")
    processed_filename = final_output_path # assign final path
    # if processing == True: #Only combine files when nothing goes wrong
    #     # COMBINE CLIPS
    #     final_output_path = os.path.join(output_dir, f"{job_id}_final_video.mp4") #Set final path
    #     combine_clips(clip_paths, final_output_path, shorts_size, fps)  # Pass shorts_size and fps

    #     print(f"Combined video saved at: {final_output_path}")
    #     processed_filename = final_output_path # assign final path
    # else:
    #     final_output_path = None # Set to NONE
    #     print("Invalid")
    #     processed_filename = None #Set to NONE

    # Delete individual clip files
    for clip_path in clip_paths:
        try:
            os.remove(clip_path)
        except Exception as e:
            print(f"Issue with deleting some paths: {e}")

    print(f"Temporary clips deleted from: {output_dir}")
    return processed_filename # Return final clip.


def combine_clips(clip_paths, output_path, shorts_size, fps):
    """Combines the given video clips into a single video."""
    if not clip_paths:
        print("No clip paths provided.")
        return

    app.logger.debug(f"Combining {len(clip_paths)} clips into {output_path}")
    shorts_width, shorts_height = shorts_size  # Unpack shorts_size tuple

    # Use 'mp4v' codec which is more widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, shorts_size)  

    if not out.isOpened():
        app.logger.error(f"Failed to open output file: {output_path}")
        return

    for i, clip_path in enumerate(clip_paths):
        app.logger.debug(f"Processing clip {i+1}/{len(clip_paths)}: {clip_path}")
        clip = cv2.VideoCapture(clip_path)
        
        if not clip.isOpened():
            app.logger.error(f"Could not open clip: {clip_path}")
            continue

        frame_count = 0
        while True:
            ret, frame = clip.read()
            if not ret:
                break
            
            # Ensure frame is in the correct size
            if frame.shape[1] != shorts_width or frame.shape[0] != shorts_height:
                frame = cv2.resize(frame, (shorts_width, shorts_height))
            
            out.write(frame)
            frame_count += 1
        
        app.logger.debug(f"Added {frame_count} frames from {clip_path}")
        clip.release()

    out.release()
    app.logger.debug(f"Combined video saved at: {output_path}")


def process_video_background(video_path, job_id, start_time, end_time, target_object):
    """Process video in background and update database"""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    app.logger.debug(f"Starting process_video_background for job_id: {job_id}") # Add
    # Process video
    try:
        final_video_path = reframe_video_to_shorts_in_clips(video_path, output_dir, job_id, start_time, end_time, target_object, app.config['YOLO_MODEL'])
        app.logger.debug(f"Finished reframe_video_to_shorts_in_clips for job_id: {job_id}, result: {final_video_path}")# Add
        # Update database with success
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()

        if final_video_path is not None:
            cursor.execute('''
                UPDATE video_history
                SET status = 'completed', processed_filename = ?
                WHERE job_id = ?
            ''', (final_video_path, job_id)) # set the file to the final path
        else:
             cursor.execute('''
                UPDATE video_history
                SET status = 'error'
                WHERE job_id = ?
            ''', (job_id,))

        conn.commit()
        conn.close()
    except Exception as e:
        # Update database with error
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE video_history
            SET status = 'error'
            WHERE job_id = ?
        ''', (job_id,))
        conn.commit()
        conn.close()
        print(f"Error processing video: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)

            # Get form data
            start_time = float(request.form.get('startTime', 0))
            end_time = float(request.form.get('endTime', 0))
            target_object = request.form.get('targetObject', 'person')  # Default to "person"

            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
            file.save(filepath)

            # Insert into database
            conn = sqlite3.connect(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO video_history (original_filename, upload_date, status, job_id, start_time, end_time, target_object)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, datetime.now(), 'processing', job_id, start_time, end_time, target_object))
            conn.commit()
            conn.close()

            # Start background processing
            thread = threading.Thread(target=process_video_background, args=(filepath, job_id, start_time, end_time, target_object))
            thread.start()

            return jsonify({'job_id': job_id, 'status': 'processing'})

        return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        print(f"Error in /upload: {e}")  # Log the error to the server console
        return jsonify({'error': str(e)}), 500  # Return a JSON error response

@app.route('/status/<job_id>')
def check_status(job_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('SELECT status, processed_filename FROM video_history WHERE job_id = ?', (job_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        status, processed_filename = result
        if status == 'completed' and processed_filename:
            return jsonify({
                'status': status,
                'output_file': processed_filename  # Return path to final video only
            })
        return jsonify({'status': status})

    return jsonify({'error': 'Job not found'}), 404

@app.route('/preview/<job_id>')
def preview_output(job_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('SELECT processed_filename FROM video_history WHERE job_id = ?', (job_id,))
    result = cursor.fetchone()
    print(result,result[0]) #Add
    conn.close()

    if result and result[0]:
        output_file = result[0]
        if  os.path.exists(output_file):
            # Return the clip as preview with proper MIME type
            return send_file(
                output_file,
                as_attachment=False,
                mimetype='video/mp4'
            )
            #292159e7-da49-4eac-b599-10d9dcec79b3

    return jsonify({'error': 'Output not found'}), 404

@app.route('/download/<job_id>')
def download_output(job_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('SELECT processed_filename FROM video_history WHERE job_id = ?', (job_id,))
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        output_file = result[0]
        if os.path.exists(output_file):
            return send_file(output_file, as_attachment=True,download_name="final_output.mp4")  # Set download name to final_output.mp4
    return jsonify({'error': 'Output not found'}), 404

@app.route('/history')
def history():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''
        SELECT job_id, original_filename, upload_date, status, start_time, end_time, target_object
        FROM video_history
        ORDER BY upload_date DESC
        LIMIT 20
    ''')
    results = cursor.fetchall()
    conn.close()

    history_data = []
    for result in results:
        history_data.append({
            'job_id': result[0],
            'original_filename': result[1],
            'upload_date': result[2],
            'status': result[3],
            'start_time': result[4],
            'end_time': result[5],
            'target_object': result[6]
        })

    return jsonify(history_data)

@app.route('/get_object_names') #Endpoint for get object
def get_object_names():
    try:
        model = YOLO(app.config['YOLO_MODEL']) #Set model
        return jsonify(model.names) #Return model list.
    except Exception as e:
        print(f"Error in /get_object_names: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
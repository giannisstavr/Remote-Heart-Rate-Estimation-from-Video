import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from collections import deque
import time
import tkinter as tk
from tkinter import simpledialog
from tensorflow.keras.applications.resnet import preprocess_input

# ------------------------------
# Config
# ------------------------------
MODEL_PATH    = "/home/giannisstavrakis/Downloads/weights-03-04/folder_03042025_2316/model.keras"
REQUIRED_SIZE = (224, 224)
MAX_FRAMES    = 30
INFER_SKIP    = 10       # run inference every 10 frames
CAL_DURATION  = 10.0     # seconds to collect for calibration

# ------------------------------
# Globals
# ------------------------------
offset       = 0.0       # dynamic offset from calibration
calibrating  = False     # are we in calibration mode?
cal_start    = None      # when calibration started
cal_list     = []        # list of raw preds during calibration

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def detect_face_bbox(frame, detector):
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets  = detector.detect_faces(rgb)
    if not dets:
        return None
    best  = max(dets, key=lambda d: d['confidence'])
    x,y,w,h = best['box']
    return (x,y,w,h)

def init_tracker(frame, bbox):
    tracker = cv2.legacy.TrackerKCF_create()
    return tracker if tracker.init(frame, bbox) else None

def track_face(tracker, frame):
    ok, bbox = tracker.update(frame)
    return (True,bbox) if ok else (False,None)

def crop_and_preprocess(frame, bbox):
    x,y,w,h = map(int, bbox)
    x,y = max(0,x), max(0,y)
    w = min(w, frame.shape[1]-x)
    h = min(h, frame.shape[0]-y)
    crop = frame[y:y+h, x:x+w]
    crop = cv2.resize(crop, REQUIRED_SIZE)
    crop = crop.astype("float32")
    return preprocess_input(crop)

def ask_ground_truth(avg_raw):
    root = tk.Tk()
    root.withdraw()
    prompt = f"Calibration complete.\nAverage raw output was {avg_raw:.2f}.\nEnter ground-truth heart rate:"
    gt = simpledialog.askfloat("Calibration", prompt, minvalue=0.0)
    root.destroy()
    return gt

def run_demo_with_calibration():
    global offset, calibrating, cal_start, cal_list

    model    = load_model()
    detector = MTCNN()
    cap      = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    face_buffer   = deque(maxlen=MAX_FRAMES)
    tracker       = None
    has_face      = False
    frame_counter = 0
    last_pred_str = None   # holds "XX.XX bpm" once computed

    window_name   = "AI Remote Heart Rate Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed. Exiting.")
            break

        disp = frame.copy()

        # --- Detection / Tracking ---
        if not has_face:
            bbox = detect_face_bbox(frame, detector)
            if bbox:
                tracker  = init_tracker(frame, bbox)
                has_face = tracker is not None
                if has_face:
                    face_buffer.clear()
                    last_pred_str = None
        else:
            ok, tb = track_face(tracker, frame)
            if ok:
                face_buffer.append(crop_and_preprocess(frame, tb))
                x,y,w,h = map(int, tb)
                # draw bounding box
                cv2.rectangle(disp, (x,y), (x+w, y+h), (0,255,0), 2)
                # ROI label centred above box
                label, font = "ROI", cv2.FONT_HERSHEY_SIMPLEX
                fs, th = 0.6, 2
                (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
                lx = x + (w - tw)//2
                ly = y - 10
                cv2.rectangle(disp,
                              (lx-5, ly-tht-5),
                              (lx+tw+5, ly+5),
                              (0,0,0), cv2.FILLED)
                cv2.putText(disp, label, (lx, ly), font, fs, (0,255,0), th)
            else:
                # lost face
                has_face      = False
                tracker       = None
                face_buffer.clear()
                last_pred_str = None

        # --- Keypress for calibration start ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and not calibrating:
            calibrating = True
            cal_start   = time.time()
            cal_list    = []
            print(">>> Calibration started for 10 s...")

        # inside your main loop, after cv2.waitKey and before frame_counter +=1:
        if key == ord('n'):
            face_buffer.clear()
            last_pred_str = None
            print(">>> Buffer cleared for new subject")


        # --- Inference & Calibration Collection ---
        raw_pred = None
        if len(face_buffer) == MAX_FRAMES and frame_counter % INFER_SKIP == 0:
            batch = np.expand_dims(np.stack(face_buffer,0),0)
            preds = model.predict(batch)[0]
            raw_pred = float(preds[0])

            if calibrating:
                cal_list.append(raw_pred)
            else:
                corr = raw_pred + offset
                last_pred_str = f"{corr:.2f} bpm"

        # --- Check end of calibration window ---
        if calibrating and (time.time() - cal_start) >= CAL_DURATION:
            avg_raw = sum(cal_list)/len(cal_list) if cal_list else 0.0
            gt      = ask_ground_truth(avg_raw)
            if gt is not None:
                offset = gt - avg_raw
                print(f">>> Offset set to {offset:.2f}")
            else:
                print(">>> Calibration cancelled.")
            calibrating = False

        # --- Unified status label logic ---
        if calibrating:
            display_label = "Calibrating..."
        elif not has_face:
            display_label = "Face not detected"
        elif len(face_buffer) < MAX_FRAMES:
            display_label = f"Buffering... ({len(face_buffer)}/{MAX_FRAMES})"
        elif last_pred_str is None:
            # buffer is full but no prediction yet
            display_label = "Calculating..."
        else:
            display_label = last_pred_str

        # draw the single high‑contrast overlay
        font       = cv2.FONT_HERSHEY_SIMPLEX
        fs, th     = 0.8, 2
        text_color = (0,255,255)
        bg_color   = (0,0,0)
        x0, y0     = 10, 40
        (w0, h0), baseline = cv2.getTextSize(display_label, font, fs, th)
        cv2.rectangle(disp,
                      (x0-5, y0-h0-5),
                      (x0+w0+5, y0+baseline+5),
                      bg_color, cv2.FILLED)
        cv2.putText(disp, display_label, (x0, y0), font, fs, text_color, th)

        # --- Show frame and handle fullscreen toggle / quit ---
        cv2.imshow(window_name, disp)
        if key == ord('q'):
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo_with_calibration()

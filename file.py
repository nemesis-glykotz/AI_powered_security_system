import cv2
import mediapipe as mp
import numpy as np
import time
import os
from deepface import DeepFace
import telegram


KNOWN_FACES_DIR = "known_faces"
DETECTION_INTERVAL = 5
LIVENESS_THRESHOLD = 0.02
LIVENESS_FRAMES_REQUIRED = 3
COOLDOWN_TIME = 6
RECOGNITION_INTERVAL = 3

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"


MEAN_BRIGHTNESS_THRESHOLD = 200
STD_BRIGHTNESS_THRESHOLD = 60
MEAN_SATURATION_THRESHOLD = 140

bot = telegram.Bot(token=BOT_TOKEN)
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
frame_count = 0
last_alert_time = 0
previous_eye_ratio = None
liveness_counter = 0
last_recognition_time = 0
previous_gray = None
motion_score = 0

print("[INFO] Preloading known face embeddings...")
try:
    files = [f.path for f in os.scandir(KNOWN_FACES_DIR) if f.is_file()]
    if files:
        DeepFace.find(img_path=files[0], db_path=KNOWN_FACES_DIR, enforce_detection=False)
        print("[INFO] Face database ready.")
    else:
        print("[WARN] No known faces found in the folder.")
except Exception as e:
    print("[ERROR] DeepFace preload failed:", e)


# ---------------- HELPERS ----------------
def calculate_eye_aspect_ratio(landmarks, left_indices, right_indices):
    left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in left_indices])
    right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in right_indices])

    def eye_ratio(eye):
        vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        return vertical / (2.0 * horizontal)

    return (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2.0


def is_screen_spoof(face_roi_bgr):

    try:
        if face_roi_bgr.size == 0:
            return False


        roi = cv2.resize(face_roi_bgr, (128, 128), interpolation=cv2.INTER_AREA)

 
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_v = np.mean(hsv[..., 2]) 
        mean_s = np.mean(hsv[..., 1]) 

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        std_v = np.std(gray)

        if mean_v >= MEAN_BRIGHTNESS_THRESHOLD:
            return True
        if std_v >= STD_BRIGHTNESS_THRESHOLD:
            return True
        if mean_s >= MEAN_SATURATION_THRESHOLD:
            return True

        return False
    except Exception as ex:
        print("[ERROR] is_screen_spoof:", ex)
        return False


def detect_liveness(rgb_frame):

    global previous_eye_ratio, liveness_counter, previous_gray, motion_score

    gray = cv2.cvtColor(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    results = face_mesh.process(rgb_frame)
    live_score = 0.0

    if previous_gray is not None:
        diff = cv2.absdiff(previous_gray, gray)
        motion_level = np.mean(diff)
        motion_score = 0.8 * motion_score + 0.2 * motion_level
        if motion_score > 2:
            live_score += 0.5
    previous_gray = gray

    if not results.multi_face_landmarks:
        liveness_counter = 0
        return False

    for face_landmarks in results.multi_face_landmarks:
        points = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
        depth_variation = np.std(points[:, 2])
        if depth_variation > 0.0008:
            live_score += 0.5

        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]
        ear = calculate_eye_aspect_ratio(face_landmarks.landmark, left_eye_idx, right_eye_idx)
        if previous_eye_ratio is None:
            previous_eye_ratio = ear
            return False

        diff = abs(previous_eye_ratio - ear)
        previous_eye_ratio = ear
        if diff > LIVENESS_THRESHOLD:
            live_score += 0.5

    brightness_std = np.std(gray)
    if brightness_std < 15:
        live_score -= 0.5

    return live_score >= 1.0


def recognize_face(image_path):
    try:
        result = DeepFace.find(img_path=image_path, db_path=KNOWN_FACES_DIR, enforce_detection=False, silent=True)
        if len(result) > 0 and not result[0].empty:
            best_match = result[0].iloc[0]
            distance = best_match.get("distance", 1.0)
            if distance < 0.32:
                print(f"[INFO] Recognized as known person (distance={distance:.2f})")
                return True
        return False
    except Exception as e:
        print("[ERROR] Recognition failed:", e)
        return False


def send_telegram_alert(image_path):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < COOLDOWN_TIME:
        return
    last_alert_time = now
    try:
        bot.send_message(chat_id=CHAT_ID, text="🚨 Intruder detected! Unknown person spotted.")
        with open(image_path, "rb") as photo:
            bot.send_photo(chat_id=CHAT_ID, photo=photo)
        print("[ALERT] Telegram notification sent.")
    except Exception as e:
        print("[ERROR] Telegram:", e)


print("🔒 AI Security System started.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("❌ Camera not accessible.")
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % DETECTION_INTERVAL == 0:
        detections = face_detection.process(rgb_frame)

        if detections.detections:
            for detection in detections.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)


                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + bw)
                y2 = min(h, y + bh)

                face_roi = frame[y1:y2, x1:x2]

                if is_screen_spoof(face_roi):
                    cv2.putText(frame, "⚠️ Spoof (screen) detected", (x1, max(10, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    print("[WARN] Immediate screen spoof detected.")


                    os.makedirs("temp", exist_ok=True)
                    img_path = os.path.join("temp", f"spoof_{int(time.time())}.jpg")
                    cv2.imwrite(img_path, frame)
                    send_telegram_alert(img_path)
                    continue

                is_live = detect_liveness(rgb_frame)
                color = (0, 255, 0) if is_live else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

                if is_live:
                    cv2.putText(frame, "Live Face Detected", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if time.time() - last_recognition_time > RECOGNITION_INTERVAL:
                        os.makedirs("temp", exist_ok=True)
                        img_path = os.path.join("temp", "capture.jpg")
                        cv2.imwrite(img_path, frame)

                        known = recognize_face(img_path)
                        last_recognition_time = time.time()

                        if not known:
                            send_telegram_alert(img_path)
                            cv2.putText(frame, "Unknown Person!", (30, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, "Authorized", (30, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Spoof Suspected", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    print("[WARN] Possible spoof attempt.")
        else:
            cv2.putText(frame, "No Face Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    cv2.imshow("AI Security System - Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 System shut down successfully.")

# AI Face Recognition Security System

An AI-powered real-time security system using OpenCV, MediaPipe, and DeepFace.  
Detects live faces, prevents spoofing, recognizes authorized users, and sends Telegram alerts for intruders.

## Features
- Real-time face detection using MediaPipe BlazeFace
- Liveness detection via blink, motion, and brightness checks
- Face recognition with DeepFace
- Telegram alert system for unknown or spoofed faces

## Requirements
- Python 3.9+
- OpenCV
- MediaPipe
- DeepFace
- Telegram Bot Token

## How to Run
1. pip install opencv-python mediapipe deepface python-telegram-bot numpy
2. Put your real `BOT_TOKEN` and `CHAT_ID` into `filename.py`.
3. Create a folder `known_faces/` and add clear frontal images (one image per person).
4. Run:

python filename.py

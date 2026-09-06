# 🚗 Driver Drowsiness Detection

A real-time **Driver Drowsiness Detection System** built using **Computer Vision and Deep Learning**.

The system uses a webcam to monitor the driver's eyes and combines **CNN-based eye-state classification** with the **Eye Aspect Ratio (EAR)** to detect signs of drowsiness.

When prolonged eye closure is detected, the system identifies the driver as **DROWSY** and triggers an alert.

---

## 📌 Project Overview

Driver fatigue and drowsiness are major causes of road accidents.

This project aims to build a computer vision system that continuously monitors a driver's eye state and detects possible drowsiness in real time.

The system combines two approaches:

- 🧠 **CNN** — Classifies the eye as Open or Closed
- 👁️ **EAR (Eye Aspect Ratio)** — Measures how open or closed the eye is
- 🎯 **Temporal Detection** — Checks whether the eyes remain closed for consecutive frames
- 🚨 **Alert System** — Warns the driver when drowsiness is detected

---

## ✨ Features

- 🎥 Real-time webcam processing
- 👤 Face detection using MediaPipe
- 👁️ Eye landmark detection
- 🧠 CNN-based eye-state classification
- 📏 Eye Aspect Ratio (EAR) calculation
- ⏱️ Consecutive-frame drowsiness detection
- 🚨 Drowsiness alert system
- 🌐 Streamlit web interface
- 📊 Live driver status
- 📈 CNN confidence and EAR information

---

## 🧠 How It Works

The system processes the webcam video frame by frame.

```text
Webcam
   ↓
Video Frame
   ↓
MediaPipe Face Mesh
   ↓
Eye Landmarks
   ↓
Eye Region Extraction
   ↓
┌─────────────────────┐
│                     │
│  CNN Classification │
│        +            │
│      EAR            │
│                     │
└─────────────────────┘
   ↓
Eye State
   ↓
Consecutive Frame Check
   ↓
Driver Status
   ↓
AWAKE / EYES CLOSED / DROWSY
   ↓
🚨 Alert

import streamlit as st
import cv2
import numpy as np

# Access the webcam (use 0 for the default camera)
cap = cv2.VideoCapture(0)

st.title("Webcam Video Stream in Streamlit")

# Create a placeholder for the video stream
frame_placeholder = st.empty()

while cap.isOpened():
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        st.write("Failed to capture video frame.")
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame in Streamlit
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

# Release the webcam
cap.release()

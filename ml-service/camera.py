import cv2
import streamlit as st
import mediapipe as mp

def start_feed():
    # Initialize the MediaPipe Face Detection model
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Open the default camera (0)
    cam = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cam.isOpened():
        st.error("Error: Could not open camera.")
        return

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

    # Create a placeholder in Streamlit for displaying images
    image_placeholder = st.empty()

    # Initialize Face Detection model
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cam.read()

            # If frame is read correctly, ret is True
            if not ret:
                st.error("Error: Failed to grab frame.")
                break

            # Convert the frame to RGB (Streamlit expects RGB format, not BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for face detection
            results = face_detection.process(frame_rgb)

            # If faces are detected, draw the face detection annotations
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame_rgb, detection)

            # Write the frame to the output file (with face detection annotations if any)
            out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            # Display the frame in Streamlit using st.image()
            image_placeholder.image(frame_rgb, channels="RGB")
from collections import deque
import cv2, mediapipe as mp, streamlit as st, os, tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as img_keras

emotions =  ("Happy", "Neutral", "Sad")
# emotions = ("Angry", "Disgusted", "Feared", "Happy", "Sad", "Surprise", "Neutral")
path = os.path.join(os.path.dirname(__file__),'emotion_model.keras')
Q = deque(maxlen=10)


def setup_cv(cam, image_placeholder):

    if not cam.isOpened():

        st.error("Error: Could not open camera.")
        return
    
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    setup_mp(cam, mp_drawing, mp_face_mesh, drawing_spec, image_placeholder)


def setup_mp(cam, mp_drawing, mp_face_mesh, drawing_spec, image_placeholder):

    counter = 0
    
    model = tf.keras.models.load_model(path, compile=False)
    
    with mp_face_mesh.FaceMesh(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as face_mesh:
    
        while cam.isOpened():
            
            suc, image = cam.read()

            if not suc:

                st.error("Error: Failed to grab frame.")

                break
            
            img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            img.flags.writeable = False

            results = face_mesh.process(img)

            img.flags.writeable = True


            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:

                for face_landmarks in results.multi_face_landmarks:

                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )

                    h, w, _ = img.shape

                    cx_min, cy_min = w, h

                    cx_max, cy_max = 0, 0


                    for lm in face_landmarks.landmark:

                        cx, cy = int(lm.x * w), int(lm.y * h)

                        cx_min, cy_min = min(cx, cx_min), min(cy, cy_min)

                        cx_max, cy_max = max(cx, cx_max), max(cy, cy_max)

                    margin = 20
                    cx_min = max(0, cx_min - margin)
                    cy_min = max(0, cy_min - margin)
                    cx_max = min(w, cx_max + margin)
                    cy_max = min(h, cy_max + margin)

                    detected_face = image[cy_min:cy_max, cx_min:cx_max]
                    detected_face = cv2.resize(detected_face, (64, 64))
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

                    img_pixels = img_keras.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255

                    emotion = model.predict(img_pixels)[0]
                    Q.append(emotion)
                    results = np.array(Q).mean(axis=0)
                    i = np.argmax(results)
                    label = emotions[i]

                    cv2.putText(img, label, (cx_min, cy_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(img, (cx_min, cy_min), (cx_max, cy_max), (0, 255, 0), 2)


                    # setup_model()

                    # exit(1)

                #     data_dir = os.path.join(os.path.dirname(__file__), '..', 'data','Sad')
                #     filename = os.path.join(data_dir, f'detected_face_{counter}.png')


                #     cv2.imwrite(filename=filename, img=detected_face)

                #     counter += 1
            
                image_placeholder.image(img, channels="BGR")

    

def setup_model():

    model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__),'emotion_model.hdf5'),compile=False)

    for layer in model.layers[:-1]: 
        layer.trainable = False

    x = model.layers[-2].output

    new_output = tf.keras.layers.Dense(3, activation='softmax')(x)

    new_model = tf.keras.models.Model(model.input, new_output)

    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(os.path.dirname(__file__),'..', 'data'),  # Path to your training dataset
        target_size=(64, 64),  # Image size that matches the input size (64x64 in this case)
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale'
    )


    new_model.fit(train_generator, epochs=100, steps_per_epoch=train_generator.samples // 32)

    # tf.keras.saving.model(new_model, os.path.join(os.path.dirname(__file__),'emotion_model.hdf5'))

    new_model.save(os.path.join(os.path.dirname(__file__),'emotion_model.keras'))

    






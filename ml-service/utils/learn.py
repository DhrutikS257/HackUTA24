import streamlit as st, cv2, shutil, os
import tensorflow as tf, time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as img_keras
from utils.feed import stop_feed
import mediapipe as mp
def learn_screen():

    if 'feed_running' not in st.session_state:
        st.session_state['feed_running'] = False


    cam = cv2.VideoCapture(0)

    image_placeholder = st.empty()

    button_container = st.container()

    

    with button_container:

        _, col2,_ = st.columns([3.5,3.5,1.6])

        with col2: 

            if st.button('Train model', key='train_model'):

                st.session_state['feed_running'] = True

                with st.spinner('Happy Face ...'):
                        
                    get_data('Happy', cam, image_placeholder)

                    st.success('Happy face data collected')

                with st.spinner('Neutral Face ...'):
                    
                    get_data('Neutral', cam, image_placeholder)

                    st.success('Neutral face data collected')

                with st.spinner('Sad Face ...'):

                    get_data('Sad', cam, image_placeholder)

                    st.success('Sad face data collected')

                stop_feed(cam)

                image_placeholder.empty()

                with st.spinner('Training the model ...'):

                    train_model()

                    st.success('Model trained successfully')



                
 

def train_model():

    orig_model = os.path.join(os.path.dirname(__file__),'emotion_model.hdf5')

    new_model = os.path.join(os.path.dirname(__file__),'emotion_model_copy.hdf5')

    shutil.copy(orig_model, new_model)

    model = tf.keras.models.load_model(new_model, compile=False)

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
        os.path.join(os.path.dirname(__file__),'..', 'data'), 
        target_size=(64, 64),  
        class_mode='categorical',
        color_mode='grayscale'
    )

    new_model.fit(train_generator, epochs=100, steps_per_epoch=train_generator.samples // 32)

    new_model.save(os.path.join(os.path.dirname(__file__),'emotion_model_copy.keras'))



def get_data(directory, cam, image_placeholder):

    counter = 0

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', directory)

    start_time = time.time()

    if not cam.isOpened():

        st.error("Error: Could not open camera.")
        return
    
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    with mp_face_mesh.FaceMesh(
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as face_mesh:

        while time.time() - start_time < 20:

            suc, image = cam.read()

            if not suc:

                st.error("Error: Failed to grab frame.")

                break
            
            img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            img.flags.writeable = False

            results = face_mesh.process(img)

            img.flags.writeable = True

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

                    file_name = os.path.join(data_dir, f'detected_face_{counter}.png')

                    cv2.imwrite(filename=file_name, img=detected_face)

                counter += 1

                image_placeholder.image(img, channels="RGB")


import streamlit as st, cv2
from utils.cv import setup_cv

def start_feed(cam, image_placeholder):

    st.session_state['feed_running'] = True

    setup_cv(cam, image_placeholder)


def stop_feed(cam):

    st.session_state['feed_running'] = False

    cam.release()

def feed_screen():

    if 'feed_running' not in st.session_state:
        st.session_state['feed_running'] = False


    cam = cv2.VideoCapture(0)

    image_placeholder = st.empty()

    button_container = st.container()


    # Start/Stop buttons inside the container to keep them at the bottom
    with button_container:
        col1, col2, col3 = st.columns([3.80, 2, 1])
        with col1:
            if st.button('Start Feed', key='start_feed'):
                start_feed(cam, image_placeholder)

        with col3:
            if st.button('Stop Feed', key='stop_feed'):
                stop_feed(cam)
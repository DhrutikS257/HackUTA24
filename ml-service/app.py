import streamlit as st
import cv2
import utils
import numpy as np
# import utils.feed as 
from utils.feed import start_feed, stop_feed

def app():

    # st.title("Emotion Damage Detector")
    st.markdown("<h1 style='text-align: center;'>EDH</h1>", unsafe_allow_html=True)


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



    # start_feed(cam)

    # st.button('Stop', key='stop', on_click= stop_feed(cam))



app()




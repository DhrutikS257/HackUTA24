import streamlit as st
import cv2, os
import utils
import numpy as np
from utils.feed import feed_screen
from utils.learn import learn_screen

def app():


    gif_path = os.path.join(os.path.dirname(__file__), 'holy_moly.gif')

    left_co, cent_co,last_co,last_last,last_last_last = st.columns(5)
    with last_co:
        st.image(gif_path)

    # st.markdown(f'<img src={os.path.join(os.path.dirname(__file__),'spin.gif')}/>', unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>EDH</h1>", unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Feed", "Train Model"])

    if page == "Train Model":
        learn_screen()

    elif page == "Feed":
        feed_screen()

    # st.sidebar.title('Navigation')
    # pg = st.sidebar.radio('Go to', ['Feed', 'Learn'])

    # if pg == "EDH Feed":
    #     feed_screen()
    # elif pg == "Other Screen":
    #     learn_screen()

    # st.markdown("<h1 style='text-align: center;'>EDH</h1>", unsafe_allow_html=True)


    # if 'feed_running' not in st.session_state:
    #     st.session_state['feed_running'] = False


    # cam = cv2.VideoCapture(0)

    # image_placeholder = st.empty()

    # button_container = st.container()


    # # Start/Stop buttons inside the container to keep them at the bottom
    # with button_container:
    #     col1, col2, col3 = st.columns([3.80, 2, 1])
    #     with col1:
    #         if st.button('Start Feed', key='start_feed'):
    #             start_feed(cam, image_placeholder)

    #     with col3:
    #         if st.button('Stop Feed', key='stop_feed'):
    #             stop_feed(cam)



    # start_feed(cam)

    # st.button('Stop', key='stop', on_click= stop_feed(cam))



app()




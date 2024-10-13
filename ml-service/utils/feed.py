import streamlit as st
from .cv import setup_cv

def start_feed(cam, image_placeholder):

    st.session_state['feed_running'] = True

    setup_cv(cam, image_placeholder)


def stop_feed(cam):

    st.session_state['feed_running'] = False

    cam.release()
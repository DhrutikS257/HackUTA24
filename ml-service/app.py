import streamlit as st
from camera import start_feed
import numpy as np

def app():

    st.title("Webcam Video Stream in Streamlit")

    start_feed()

app()

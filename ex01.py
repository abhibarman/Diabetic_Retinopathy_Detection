import streamlit as st
import time

progress_bar = st.progress(0)

for i in range(100):
    # Perform some computation or task
    time.sleep(0.1)
    
    # Update the progress bar
    progress_bar.progress(i + 1)

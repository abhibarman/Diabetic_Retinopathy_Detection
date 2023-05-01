import streamlit as st

button = st.button("Click me")
button_css = """
    <style>
        div.stButton > button:first-child {
            background-color: #345beb;
            color: white;
        }
    </style>
"""
st.markdown(button_css, unsafe_allow_html=True)

import streamlit as st
from helper_functions import *  



st.set_page_config(page_title="Sentiment Analysis", layout="wide")
html_temp = """
  <div style="background-color: rgba(0, 0, 0, 0.7); 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8);
                text-align: center;">
        <h2 style="color: white; font-size: 30px; margin: 0; 
                   text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);">
                   Comprehensive Data Science Toolkit for Sentiment Analysis Tasks
        </h2>
    </div>
""" 
st.markdown(html_temp, unsafe_allow_html=True)

welcome_message = """
    <div style="text-align: center; margin-top: 20px;">
        <h6 style="color: #000000; font-size: 20px; 
                   font-family: 'Arial', sans-serif; 
                   text-shadow: 1px 1px 5px rgba(0.1, 0.1, 0.1, 0.6); 
                   padding: 10px;">
                   Explore our features and enjoy your experience!
        </h6>
    </div>
"""
st.markdown(welcome_message, unsafe_allow_html=True)

# Title and description
#st.title("Customer Product Reviews Sentiment Analysis App")
# app design
set_bg_hack('Picture1.png')


# Create a container for the image
with st.container():
    # Add the image inside the container with specified width and height
    st.image(
        "Home.png",  # Replace with the path to your image
        caption="Home Page",
        use_container_width=False,  # Set to False to use custom width and height
    )


st.sidebar.header("About Application")
st.sidebar.info("A Customer Sentiment analysis App which collect reviews any area . \
                The different Visualizations will help us to get overall exploration of reviews.\
                then determine the Sentiments of those reviews")
#st.sidebar.text("Built with Streamlit")

st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("medotalaat20177@gmail.com")
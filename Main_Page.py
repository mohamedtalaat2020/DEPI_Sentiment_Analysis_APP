import streamlit as st
from helper_functions import *  

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# HTML and CSS for the hover effect and main content
html_temp = """
<style>
.hover-div:hover {
    background-color: rgba(0, 0, 255, 0.1);
    box-shadow: 0 10px 50px rgba(255, 0, 255, 0.8);
}
</style>
<div class="hover-div" style="background-color: rgba(0, 0, 0, 0.7); 
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

# Set background image
set_bg_hack('Picture1.png')

# Add an image with a styled caption
st.image(
    "Home.png",  # Replace with the path to your image
    caption="Home Page",
    use_container_width=True  # Automatically adjusts the image width to the container size
)

# Sidebar content
st.sidebar.header("About App")
st.sidebar.info("""
A Customer Sentiment analysis Project which collects data of reviews of Amazon products. 
The reviews will then be used to determine the Sentiments of those reviews. 
The different Visualizations will help us get a feel of the overall exploration of reviews.
""")
st.sidebar.text("Built with Streamlit")

st.sidebar.header("For Any Queries/Suggestions Please reach out at:")
st.sidebar.info("medotalaat20177@gmail.com")

# Main content
st.title("Customer Product Reviews Sentiment Analysis App")
st.write("""
Welcome to the Customer Product Reviews Sentiment Analysis App! This tool helps you analyze customer reviews from Amazon products to determine their sentiments. 
Explore various visualizations to gain insights into customer feedback.
""")
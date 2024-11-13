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


# Title and description
#st.title("Customer Product Reviews Sentiment Analysis App")
# app design
set_bg_hack('Picture1.png')
st.write("## Team Members 🧑‍💻")

st.markdown("""
<style>
.team-members {
  background-color: #f2f2f2;
  padding: 20px;
  border-radius: 10px;
}

h2 {
  color: #333;
  text-align: center;
}

ul {
  list-style-type: none;
  padding: 0;
  margin: 0;
}

li {
  margin-bottom: 10px;
  font-size: 18px;
  font-weight: bold;
  color: #555;
}
</style>

<div class="team-members">
  <ul>
    <li>Mohamed Talaat Abo Elftouh</li>
    <li>Mohamed Mostafa Abdelhamed</li>
    <li>Moaz Mohamed Tawfik</li>
    <li>Amr Khaled Mostafa</li>
    <li>Mahmoud Mohammed Abdelmawgoud</li>
    <li>Mohamed Alaa Elsayad</li>
  </ul>
</div>
""", unsafe_allow_html=True)


st.sidebar.header("About App")
st.sidebar.info("A Customer Sentiment analysis Project which collect data of reviews of Amazon products. The reviews will then be used to determine the Sentiments of those reviews. \
                The different Visualizations will help us get a feel of the overall exploration of reviews")
st.sidebar.text("Built with Streamlit")

st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("medotalaat20177@gmail.com")
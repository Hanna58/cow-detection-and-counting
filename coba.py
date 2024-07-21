import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io
import os

# Initialize Roboflow with your API key
rf = Roboflow(api_key="WHgmsmbdXkhuV1VAw4BK")
project = rf.workspace("hanna-fg2n9").project("cattle-detection-and-counting")
model = project.version(9).model

def home_page():
    st.title("Cow Detection and Counting Application")
    st.write("""
    This web application allows you to upload an image and detect cows in it using a pre-trained model. 
    The home page provides an overview of the application's functionality, while the upload page lets you 
    upload an image and view the detection results.
    """)

def upload_page():
    st.title("Upload Image for Cow Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting cows in the image...")

        # Save the uploaded image to a temporary file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict using Roboflow model
        result = model.predict("temp_image.jpg", confidence=40, overlap=30).json()

        # Visualize and display the prediction
        result_image_path = "prediction.jpg"
        model.predict("temp_image.jpg", confidence=40, overlap=30).save(result_image_path)
        st.image(result_image_path, caption='Prediction Result.', use_column_width=True)

        # Extract and display the number of detected cows
        num_cows = len(result['predictions'])
        st.write(f"Number of cows detected: {num_cows}")

        # Display the prediction results
        # st.write(result)

        # Clean up temporary file
        os.remove("temp_image.jpg")

# Main application structure
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Upload"])

if selection == "Home":
    home_page()
elif selection == "Upload":
    upload_page()

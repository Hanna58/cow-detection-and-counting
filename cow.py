import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io
import os
import pandas as pd

# Initialize Roboflow with your API key
try:
    rf = Roboflow(api_key="WHgmsmbdXkhuV1VAw4BK")
    project = rf.workspace("hanna-fg2n9").project("cattle-detection-and-counting")
    model = project.version(9).model
except Exception as e:
    st.error(f"Error initializing Roboflow: {e}")
    st.stop()

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
        st.write("Detecting cows in the image...")

        # Save the uploaded image to a temporary file
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Predict using Roboflow model
            result = model.predict(temp_image_path, confidence=40, overlap=30).json()

            # Visualize and display the prediction
            result_image_path = "prediction.jpg"
            model.predict(temp_image_path, confidence=40, overlap=30).save(result_image_path)
            st.image(result_image_path, caption='Prediction Result.', use_column_width=True)

            # Extract and display the number of detected cows
            num_cows = len(result['predictions'])
            st.write(f"Number of cows detected: {num_cows}")

            # Calculate and display average confidence score
            if num_cows > 0:
                confidences = [pred.get('confidence', 0) for pred in result['predictions']]
                avg_confidence = sum(confidences) / num_cows
                st.write(f"Average confidence score: {avg_confidence:.2f}")

                # Create and display a summary table
                df = pd.DataFrame(result['predictions'])
                st.write("Prediction Summary Table:")
                # Ensure all relevant columns are included
                columns_to_show = ['class', 'confidence', 'x', 'y', 'width', 'height']
                # Display only columns that exist in the DataFrame
                columns_to_show = [col for col in columns_to_show if col in df.columns]
                summary_df = df[columns_to_show]
                st.dataframe(summary_df)
            else:
                st.write("No cows detected.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

        # Clean up temporary file
        os.remove(temp_image_path)

# Main application structure
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Upload"])

if selection == "Home":
    home_page()
elif selection == "Upload":
    upload_page()

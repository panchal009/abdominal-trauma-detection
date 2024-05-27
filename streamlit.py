import streamlit as st
import keras
from keras.preprocessing import image
import numpy as np
import keras_cv

# Load the trained model
model = keras.models.load_model('abdominal_trauma_model.h5')

# Function to make predictions
def predict(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust target_size as per your model's requirement
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis

    predictions = model.predict(img_array)
    return predictions

# Streamlit app layout
st.title("Abdominal Trauma Detection")
st.write("Upload an image to detect trauma")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image_path = f"./{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(image_path, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predictions = predict(image_path)
    st.write(predictions)

    # Assuming binary classification with two classes: Trauma and No Trauma
    if predictions[0][0] > 0.5:
        st.write("Prediction: Trauma Detected")
    else:
        st.write("Prediction: No Trauma Detected")

# %%writefile tb-school/app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Suppress warnings if you'd like
# import warnings
# warnings.filterwarnings('ignore')

# Load your pre-trained model
model = tf.keras.models.load_model('./model.hdf5')

# Function to preprocess the image and make predictions
def lung_defect(img):

    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Preprocessing the image to fit the model input shape
    img = tf.image.resize(img, [128, 128])
    img = img[None, ...]
    # img_array = img_array / 255.0  # Assuming the model expects the input in this range
    # img_array = img_array.reshape((1, 224, 224, 3))  # Adjusting to the input shape

    # Make a prediction
    prediction = model.predict(img).tolist()[0]
    class_names = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']

    # Returning a dictionary of class names and corresponding predictions
    return {class_names[i]: float(prediction[i]) for i in range(4)}

# Streamlit user interface
st.title('Lung Defect Classification')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    predictions = lung_defect(image)

    # Display the predictions as a bar chart
    st.bar_chart(predictions)

# .run()
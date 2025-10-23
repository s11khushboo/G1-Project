import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained CNN model
@st.cache_resource
def load_cnn_model():
    model = load_model('ic_cnn_model.keras')  # replace with your model path
    return model

model = load_cnn_model()

# with open("index.html", "r") as file:
#     html_content = file.read()
# st.markdown(html_content, unsafe_allow_html=True)
# Title
st.title("CNN Image Classification App")
st.write("Upload an image and let the model predict its class!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img = img.resize((32, 32))  # replace with your model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if your model was trained with normalized images

    # Predict
  

    pred_probs = model.predict(img_array)

    # Get predicted class
    pred_class = np.argmax(pred_probs, axis=1)[0]

    # Map to class label if you have class_names
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    max_class = class_names[np.argmax(pred_probs[0])]
    max_prob = np.max(pred_probs[0]) * 100
    print(f"Predicted: {max_class} ({max_prob:.2f}%)")
    pred_label = class_names[pred_class]

    # Display result
    st.write(f"Predicted Class: {pred_label} accuracy:{max_prob}")

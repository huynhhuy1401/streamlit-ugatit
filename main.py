import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st


def load_image(path):
  image_raw = tf.io.read_file(path)
  image = tf.image.decode_image(image_raw, channels=3)
  return image

def resize(image):
    resized_image =  tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_image = tf.cast(resized_image, tf.float32)
    resized_image = tf.expand_dims(resized_image, 0)

    return resized_image

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def transform(image):
    test_image_resized = resize(image)


    with tf.io.gfile.GFile('selfie2anime.tflite', 'rb') as f:
        model_content = f.read()

    # Initialze TensorFlow Lite inpterpreter.
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    # Set model input
    interpreter.set_tensor(input_index, test_image_resized)

    # Run inference
    interpreter.invoke()



    test_out = normalize(output()[0]) * 255.0
    test_out = test_out.astype(np.uint8)

    return test_out

st.header("Generate anime picture from your selfie")
st.write("Choose any image and get corresponding anime art:")

uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    test_image_resized = resize(image)
    out = transform(image)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original', use_column_width='always')
    with col2:
        st.image(out, caption='Transformed', use_column_width='always')

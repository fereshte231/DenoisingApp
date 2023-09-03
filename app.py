import streamlit as st
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Add, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout, \
    Activation, Subtract
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
import os
import pydicom
import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Subtract
from tensorflow.keras.layers import Input
import glob
from PIL import Image
import os
import base64

currentDir = os.path.abspath(os.path.dirname(__name__))


# Function to get DNCNN model
def get_dncnn_model(input_channel_num):
    inpt = Input(shape=(None, None, input_channel_num))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Subtract()([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x)
    return model


# Function to get the desired model
def get_model(input_channel_num, model_name="srresnet"):
    if model_name == "srresnet":
        return get_srresnet_model(input_channel_num=input_channel_num)
    elif model_name == "unet":
        return get_unet_model(input_channel_num=input_channel_num, out_ch=input_channel_num)
    elif model_name == "dncnn":
        return get_dncnn_model(input_channel_num=input_channel_num)
    else:
        raise ValueError("model_name should be 'srresnet' or 'unet'")


# Function to load and preprocess DICOM image
def load_dicom_image(file):
    ds = pydicom.dcmread(file)
    img = (ds.pixel_array)[160:890, 160:890]
    img = np.clip(img / factor, 0, 255)
    img = np.uint8(img)
    return img


def save_image(img, filename):
    cv2.imwrite(filename, img)


factor = 7

st.set_page_config(page_title="Image Denoising App", page_icon="📁")


# Streamlit web app UI
def main():
    # Load the logo image
    global currentDir
    logo_image = Image.open(currentDir + '/Untitled.png')
    st.sidebar.image(logo_image, caption='', use_column_width=True)
    logo_image = Image.open(currentDir + '/dfd.PNG')
    st.sidebar.image(logo_image, caption='Denoising CT images', use_column_width=True)

    st.title("Image Denoising App")
    st.markdown(
        "<h3 style='font-size: 20px; text-align: center;'>This app is designed for denoising CT images with low dose. You can import noisy DICOM images for denoising</h3>",
        unsafe_allow_html=True)

    # Upload DICOM image 1
    uploaded_file1 = st.file_uploader("Upload the Image of the Phantom DICOM")

    if uploaded_file1 is not None:
        # Load and preprocess uploaded image 1
        img1 = load_dicom_image(uploaded_file1)

        # Load and preprocess model
        weight_file1 = "center_weights.011-50.997-29.16076_dncnn_CD_fantom.hdf5"
        input_channel_num = 1
        model1 = get_model(input_channel_num=input_channel_num, model_name="dncnn")
        model1.load_weights(weight_file1)

        # Denoise image 1
        pred1 = model1.predict(np.expand_dims(np.expand_dims(img1, 2), 0))
        denoised_image1 = np.clip(pred1[0][:, :, 0], 0, 255).astype(dtype=np.uint8)

        # Display original and denoised image 1
        col1, col2 = st.columns(2)
        with col1:
            stage1_display = st.subheader("Original Image 1")
            stage1_display.image(img1, caption="Original Image 1", use_column_width=True)
        with col2:
            stage2_display = st.subheader("Denoised Image 1")
            stage2_display.image(denoised_image1, caption="Denoised Image 1", use_column_width=True)

        # Save image 1 button
        if st.button("Save Denoised Image 1"):
            address = st.text_input("Enter the address to save the image:")

            if len(address) > 0:
                save_image(denoised_image1, address + "/denoised_image1.png")
                st.success("Denoised image 1 saved successfully!")
            else:
                st.warning("Please enter a valid address!")

    # Upload DICOM image 2
    uploaded_file2 = st.file_uploader("Upload a Human Leg DICOM Image")

    if uploaded_file2 is not None:
        # Load and preprocess uploaded image 2
        img2 = load_dicom_image(uploaded_file2)

        # Load and preprocess model
        weight_file2 = "center_weights.062-0.000-139.40607_dncnn_CD_fantom.hdf5"
        input_channel_num = 1
        model2 = get_model(input_channel_num=input_channel_num, model_name="dncnn")
        model2.load_weights(weight_file2)

        # Denoise image 2
        pred2 = model2.predict(np.expand_dims(np.expand_dims(img2, 2), 0))
        denoised_image2 = np.clip(pred2[0][:, :, 0], 0, 255).astype(dtype=np.uint8)

        # Display original and denoised image 2
        col3, col4 = st.columns(2)
        with col3:
            stage3_display = st.subheader("Original Image 2")
            stage3_display.image(img2, caption="Original Image 2", use_column_width=True)
        with col4:
            stage4_display = st.subheader("Denoised Image 2")
            stage4_display.image(denoised_image2, caption="Denoised Image 2", use_column_width=True)

        # Save image 2 button
        if st.button("Save Denoised Image 2"):
            address = st.text_input("Enter the address to save the image:")

            if len(address) > 0:
                save_image(denoised_image2, address + "/denoised_image2.png")
                st.success("Denoised image 2 saved successfully!")
            else:
                st.warning("Please enter a valid address!")

if __name__ == "__main__":
    main()

# Add footer
st.sidebar.write('Developed by QMISG')
# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gn7yavKMJM9j41Xcaluhs84e8W0kmiSh
"""
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from st_clickable_images import clickable_images



st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Brain Tumor Detection','Diabetes Prediction'],
                           menu_icon='hospital-fill',
                           icons=['🧠','activity'],
                           default_index=0)





if selected=="Brain Tumor Detection":
    class CNN_BrainMRI(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 16, 2, 1)
            self.mxpool = nn.MaxPool2d(2, 2)
            self.flat = nn.Flatten()
            self.conv_total = nn.Sequential(
                self.conv1,
                self.mxpool,
                self.conv2,
                self.mxpool
            )
            self.linear1 = nn.Linear(15376, 64)  # Adjust this based on your model's architecture
            self.linear2 = nn.Linear(64, 2)

        def forward(self, x):
            x = F.relu(self.conv_total(x))
            x = self.flat(x)
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return F.log_softmax(x, dim=1)
    
    # Check for device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(device)

    # Load model
    model = CNN_BrainMRI().to(device)
    model.load_state_dict(torch.load("CNN_BrainMRI_tumor_classification1.pt", map_location=device))
    model.eval()

    #Loading and Processing the image
    def load_and_preprocess_image(image_path):
        image = Image.open(image_path).convert("L")
        image = image.resize((128, 128))
        image_array = np.asarray(image)
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return image_tensor.to(device)

    #Predicting the Result using processed image
    def predict_tumor_presence(model, image_tensor):
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return prediction

    #ignoring the ERROR from loading the image
    warnings.filterwarnings("ignore")

    #designing the front look
    st.title("Brain Tumor Detection")

    # Test with a single image
    single_image_path = st.file_uploader("Choose an MRI Image of Brain.........", type=["jpg", "jpeg", "png"])
    if single_image_path is not None:
        left_co, cent_co1,cent_co2,cent_co3,last_co = st.columns(5)
        with cent_co1:
            image = Image.open(single_image_path)  # Open the uploaded image using PIL(pillow library)
            st.image(image, caption="Image uploded Successfully",  width=500)

        #Function call for load and predict image 
        image_tensor = load_and_preprocess_image(single_image_path)
        prediction = predict_tumor_presence(model, image_tensor)
    
        #Button for the result
        if(st.button("Click For Result")):
            if prediction==0:
                st.success("""Based on the MRI Scanned Image the patient don't have any kind of Tumor
                THE PATIENT DOES'NT HAVE BRAIN TUMOR""")
            elif prediction==1:
                st.success("""Based on the MRI Scanned Image the patient do have a tumor
                THE PATIENT DO HAVE BRAIN TUMOR""")
    else:
        st.write("Import the valid MRI Scanned image of brain witout any unwanted weights occurs!...")





if selected == "Diabetes Prediction":
    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies',max_value=15)

    with col2:
        Glucose = st.number_input('Glucose Level')

    with col3:
        BloodPressure = st.number_input('Blood Pressure value')

    with col1:
        SkinThickness = st.number_input('Skin Thickness value')

    with col2:
        Insulin = st.number_input('Insulin Level')

    with col3:
        BMI = st.number_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.number_input('Age of the Person')
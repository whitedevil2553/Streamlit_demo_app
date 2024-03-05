# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import requests
# import numpy as np
# import matplotlib.pyplot as plt

# # Function to load model
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model_url = "https://drive.google.com/uc?export=download&id=1lEiIeM4SyCsJYjfHYjyMuGlayP2yhj3Y"
#     model_file = requests.get(model_url)
#     with open("model.pt", "wb") as f:
#         f.write(model_file.content)
#     return tf.keras.models.load_model("model.pt")

# # Load the model
# model = load_model()

# # Function to make predictions
# def predict(image):
#     # Preprocess the image (if required)
#     # For example, resize the image to the input size expected by the model
#     processed_image = preprocess_image(image)
#     # Make predictions
#     prediction = model.predict(processed_image)
#     return prediction

# # Streamlit app code
# st.title("Image Classification App")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     # Make predictions
#     prediction = predict(image)
    
#     # Display the prediction
#     st.write("Prediction:", prediction)











# # Function to load and preprocess a single MRI image
# def load_and_preprocess_image(image_path):
#     image = Image.open(image_path).convert("L")  # Convert to grayscale
#     image = image.resize((128, 128))  # Resize to match model input size
#     image_array = np.asarray(image)  # Convert to numpy array
#     image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
#     return image_tensor

# # Function to predict tumor presence in a single MRI image
# def predict_tumor_presence(model, image_tensor):
#     model.eval()  # Set model to evaluation mode
#     with torch.no_grad():  # Disable gradient calculation
#         output = model(image_tensor.to(device))  # Forward pass
#         prediction = torch.argmax(output, dim=1).item()  # Get predicted class index
#     return prediction

# # Function to display prediction
# def display_prediction(image_path, prediction):
#     mapping = {0: 'No tumor', 1: 'Tumor'}
#     plt.imshow(Image.open(image_path), cmap='gray')
#     plt.title(f'Predicted: {mapping[prediction]}')
#     plt.axis('off')
#     plt.show()

# # Path to the single MRI image you want to test
# # single_image_path = "Y104.jpg"

# # Load and preprocess the single MRI image
# image_tensor = load_and_preprocess_image(uploaded_file)

# # Predict tumor presence
# prediction = predict_tumor_presence(model, image_tensor)

# # Display prediction
# display_prediction(uploaded_file, prediction)




import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
import numpy as np

# Function to load model
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://drive.google.com/uc?export=download&id=10_CL4rGvZXD0Mb1Hf0h2dA0X5LyZ5noI"
    model_file = requests.get(model_url)
    with open("model.h5", "wb") as f:
        f.write(model_file.content)
    return tf.keras.models.load_model("model.h5")

# Load the model
model = load_model()

# Function to preprocess image
def preprocess_image(image):
    # Resize the image to match model input size
    resized_image = image.resize((128, 128))
    # Convert image to numpy array and normalize pixel values
    processed_image = np.array(resized_image) / 255.0
    # Add batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Function to make predictions
def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make predictions
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app code
st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make predictions
    prediction = predict(image)
    
    # Display the prediction
    st.write("Prediction:", prediction)


















# import streamlit as st
# from streamlit_option_menu import option_menu
# from PIL import Image
# from google_drive_api import upload_to_google_drive


# # Function to perform diabetes prediction
# def diabetes_prediction():
#     # Page title
#     st.title('Diabetes Prediction using ML')

#     # Getting the input data from the user
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         Pregnancies = st.text_input('Number of Pregnancies')

#     with col2:
#         Glucose = st.text_input('Glucose Level')

#     with col3:
#         BloodPressure = st.text_input('Blood Pressure value')

#     with col1:
#         SkinThickness = st.text_input('Skin Thickness value')

#     with col2:
#         Insulin = st.text_input('Insulin Level')

#     with col3:
#         BMI = st.text_input('BMI value')

#     with col1:
#         DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

#     with col2:
#         Age = st.text_input('Age of the Person')

# # Function to perform heart disease prediction
# def heart_disease_prediction():
#     st.write("Heart Disease Prediction function goes here")

# # Streamlit app code
# if __name__ == '__main__':
#     st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

#     # Sidebar menu for disease prediction options
#     with st.sidebar:
#         selected = option_menu('Multiple Disease Prediction System',
#                                ['Diabetes Prediction',
#                                 'Heart Disease Prediction'],
#                                menu_icon='hospital-fill',
#                                icons=['activity', 'heart'],
#                                default_index=0)

#     # Render selected prediction page
#     if selected == 'Diabetes Prediction':
#         diabetes_prediction()
#     elif selected == 'Heart Disease Prediction':
#         heart_disease_prediction()

#    # Function to upload image to Google Drive
#     def upload_and_store_image(uploaded_file):
#         # Upload the image to Google Drive
#         success = upload_to_google_drive(uploaded_file)
    
#         # Display success or error message
#         if success:
#             st.success("Image uploaded to Google Drive successfully!")
#         else:
#             st.error("Failed to upload image to Google Drive. Please try again later.")

#     # Streamlit app code
#     if __name__ == '__main__':
#         uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
#         if uploaded_file is not None:
#             upload_and_store_image(uploaded_file)















# from pathlib import Path
# import streamlit as st
# from streamlit_option_menu import option_menu
# from PIL import Image


# st.set_page_config(page_title="Health Assistant",
#                    layout="wide",
#                    page_icon="🧑‍⚕️")



# with st.sidebar:
#     selected = option_menu('Multiple Disease Prediction System',

#                            ['Diabetes Prediction',
#                             'Heart Disease Prediction',
#                             'Parkinsons Prediction',
#                             'pitutary prediction',
#                             'giloma detection',
#                             'heart issues prediction'],
#                            menu_icon='hospital-fill',
#                            icons=['activity', 'heart', 'person','heart'],
#                            default_index=0)




# # Diabetes Prediction Page
# if selected == 'Diabetes Prediction':

#     # page title
#     st.title('Diabetes Prediction using ML')

#     # getting the input data from the user
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         Pregnancies = st.text_input('Number of Pregnancies')

#     with col2:
#         Glucose = st.text_input('Glucose Level')

#     with col3:
#         BloodPressure = st.text_input('Blood Pressure value')

#     with col1:
#         SkinThickness = st.text_input('Skin Thickness value')

#     with col2:
#         Insulin = st.text_input('Insulin Level')

#     with col3:
#         BMI = st.text_input('BMI value')

#     with col1:
#         DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

#     with col2:
#         Age = st.text_input('Age of the Person')

    
# if selected == 'Heart Disease Prediction':
    

#     # # drive.mount('/content/gdrive')  # Replace '/content/gdrive' with your desired mount point
#     # st.title("Image Classification with Streamlit and Google Colab")
#     # st.write("Upload an image and it will be processed in Google Colab.")
#     # uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    
#     # if uploaded_file is not None:
#     #     image = Image.open(uploaded_file)  # Open the uploaded image using PIL(pillow library)
#     #     st.image(image, caption="Uploaded Image", use_column_width=True)
#     #     # Save the uploaded image to Google Drive
#     #     with open('/content/gdrive/My Drive/Drive-image/' + uploaded_file.name, 'wb') as f:
#     #         f.write(uploaded_file.read())
#     #     st.success("Image uploaded successfully!")






#     # # Function to upload image to Google Drive
#     # def upload_to_drive(uploaded_file):
#     #     # Define the directory path in Google Drive
#     #     drive_path = Path('/content/gdrive/My Drive/Drive-image/')
    
#     #     # Create directory if it doesn't exist
#     #     drive_path.mkdir(parents=True, exist_ok=True)
    
#     #     # Get the file name
#     #     filename = uploaded_file.name
    
#     #     # Construct the full file path
#     #     file_path = drive_path / filename
    
#     #     # Write the uploaded file to Google Drive
#     #     with open(file_path, 'wb') as f:
#     #         f.write(uploaded_file.getvalue())
    
#     #     # Inform the user about successful upload
#     #     st.success("File uploaded to Google Drive successfully.")

#     # # Streamlit app code
#     # if __name__ == '__main__':
#     #     uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
#     #     if uploaded_file is not None:
#     #         upload_to_drive(uploaded_file)











#     def upload_to_drive(uploaded_file):
#         # Define the directory path in Google Drive
#         drive_path = Path('/content/gdrive/My Drive/Drive-image/')
    
#         # Check if directory exists, if not, create it
#         if not drive_path.exists():
#             drive_path.mkdir(parents=True, exist_ok=True)
    
#         # Get the file name
#         filename = Path(uploaded_file.name).name
    
#         # Construct the full file path
#         file_path = drive_path / filename
    
#         # Write the uploaded file to Google Drive
#         with open(file_path, 'wb') as f:
#             f.write(uploaded_file.getvalue())
    
#         # Inform the user about successful upload
#         st.success(f"File '{filename}' uploaded to Google Drive successfully.")

#     # Streamlit app code
#     if __name__ == '__main__':
#         uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
#         if uploaded_file is not None:
#             upload_to_drive(uploaded_file)








# import os
# from pathlib import Path
# import streamlit as st
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request

# # Google Drive API scopes
# SCOPES = ['https://www.googleapis.com/auth/drive']

# # Function to authenticate Google Drive API
# def authenticate():
#     creds = None
#     if os.path.exists('token.json'):
#         creds = Credentials.from_authorized_user_file('token.json')
#     # If credentials are not available or are invalid, authenticate
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open('token.json', 'w') as token:
#             token.write(creds.to_json())
#     return creds

# # Function to upload image to Google Drive
# def upload_to_drive(uploaded_file):
#     creds = authenticate()
#     service = build('drive', 'v3', credentials=creds)

#     # Define the folder ID where you want to upload the image
#     folder_id = 'https://drive.google.com/drive/folders/1Mkv8nm0rMpgiTJOvOpAjliz_jpp4mxWj?usp=sharing'

#     # Create metadata for the file
#     file_metadata = {
#         'name': uploaded_file.name,
#         'parents': [folder_id]
#     }

#     # Upload the file to Google Drive
#     media = MediaIoBaseUpload(uploaded_file, mimetype='image/jpeg')
#     file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    
#     # Inform the user about successful upload
#     st.success(f"File '{uploaded_file.name}' uploaded to Google Drive successfully.")

# # Streamlit app code
# def main():
#     st.title("Upload Image to Google Drive")

#     # File uploader
#     uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

#     if uploaded_file is not None:
#         upload_to_drive(uploaded_file)

# if __name__ == "__main__":
#     main()

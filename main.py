from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image


st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")



with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'pitutary prediction',
                            'giloma detection',
                            'heart issues prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person','heart'],
                           default_index=0)




# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    
if selected == 'Heart Disease Prediction':
    

    # # drive.mount('/content/gdrive')  # Replace '/content/gdrive' with your desired mount point
    # st.title("Image Classification with Streamlit and Google Colab")
    # st.write("Upload an image and it will be processed in Google Colab.")
    # uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    
    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)  # Open the uploaded image using PIL(pillow library)
    #     st.image(image, caption="Uploaded Image", use_column_width=True)
    #     # Save the uploaded image to Google Drive
    #     with open('/content/gdrive/My Drive/Drive-image/' + uploaded_file.name, 'wb') as f:
    #         f.write(uploaded_file.read())
    #     st.success("Image uploaded successfully!")






    # Function to upload image to Google Drive
    def upload_to_drive(uploaded_file):
        # Define the directory path in Google Drive
        drive_path = Path('/content/gdrive/My Drive/Drive-image/')
    
        # Create directory if it doesn't exist
        drive_path.mkdir(parents=True, exist_ok=True)
    
        # Get the file name
        filename = uploaded_file.name
    
        # Construct the full file path
        file_path = drive_path / filename
    
        # Write the uploaded file to Google Drive
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
    
        # Inform the user about successful upload
        st.success("File uploaded to Google Drive successfully.")

    # Streamlit app code
    if __name__ == '__main__':
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
        if uploaded_file is not None:
            upload_to_drive(uploaded_file)



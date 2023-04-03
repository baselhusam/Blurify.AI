import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
from streamlit_lottie import st_lottie
from ultralytics import YOLO


# Define a function to load the Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



# Set the page layout to wide
st.set_page_config(page_title = "Blurify.AI", page_icon = "🩺")
col1, col2, col3 = st.columns([0.1, 1, 0.1])
coll, colc, colr = st.columns([0.8, 1.2, 0.8])



with col2:
    # Set the title
    st.markdown("<h1 style='text-align: center'>Blurify.AI Application</h1>", unsafe_allow_html=True)


with colc:
    # Load the Lottie animation
    anim = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_wK2ITq.json")
    st_lottie(anim, key='User Interface Animation', speed=1, height=300, width=300,)



cola, colb, colc = st.columns([0.008,1.1,0.008])
colx, coly, colz = st.columns(3)


with colb:

    # Set the description
    st.markdown(""" 
            **Blurfiy.AI** is a web-based application that allows you to blur faces in an image without affecting 
            the other parts of it using **YOLO** and **HaarCascade Classifier** models to detect the faces in the image. 
            You can upload an image from your computer and select the model you want to make the detection. 
            Finally, click the "**Blur the faces**" button to blur the faces in the image.
            """)
    

    # Select the model with the uploaded image
    selected_model = st.selectbox("Select a model", ("YOLOv8", "HaarCascade Classifier"))
    st.markdown("<br>", unsafe_allow_html=True)
    upd_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    # Check if the image is uploaded
    if upd_img is not None:
        image = cv2.imdecode(np.frombuffer(upd_img.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            cv2.imwrite(temp.name, image)
            image_path = temp.name
    else:
        st.warning("Please upload an image to begin.")

    

    with coly:

        # Check if the image is uploaded
        if upd_img is not None:
            # Blur the faces button
            blur = st.button("Blur the faces")


if upd_img is not None:
    
    # if the blur button is clicked
    if blur:
            
        # Yolo detection
        if selected_model == "YOLOv8":

            weights_path = ".\\pretrained_models\\YOLO_Model.pt"
            model = YOLO('yolov8n.pt')
            model = YOLO("D:\\Projects\\ProgressSoft_Assignment\\Problem1\\pretrained_models\\YOLO_Model.pt")

            results = model(image_path)
            img = cv2.imread(image_path)

            for result in results:
                boxes = result.boxes
                if boxes:
                    for box in boxes:
                        bouding_cordinate = box.xyxy
                        x1, y1, x2, y2 = ( int(bouding_cordinate[0,0].item()), int(bouding_cordinate[0,1].item()), 
                                            int(bouding_cordinate[0,2].item()), int(bouding_cordinate[0,3].item()) )
                        x1, y1, x2, y2 = max(x1,0), max(y1, 0), max(x2, 0), max(y2, 0)
                        region = img[y1:y2, x1:x2]
                        blurred_region = cv2.GaussianBlur(region, (25,25), 100)
                        img[y1:y2, x1:x2] = blurred_region

            # Set columns
            col1, col2 = st.columns([1, 1])

            # Display Images
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(img, caption="Blurred Image", use_column_width=True)

            st.markdown("**NOTE:** you can download the blurred image by right clicking on it and selecting **Save image as....** ")


        # HaarCascade Classifier detection
        elif selected_model == "HaarCascade Classifier":

            cascade_classifier_path = ".\\pretrained_models\\haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_classifier_path)
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
            img = cv2.imread(image_path)
            for (x, y, w, h) in faces:
                region = img[y:y+h, x:x+w]
                blurred_region = cv2.GaussianBlur(region, (25,25), 75)
                img[y:y+h, x:x+w] = blurred_region
                    
            # Set columns
            col1, col2 = st.columns([1, 1])

            # Display Images
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(img, caption="Blurred Image", use_column_width=True)

            st.markdown("**NOTE:** you can download the blurred image by right clicking on it and selecting **Save image as....** ")


import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live
from streamlit_extras.app_logo import add_logo
from streamlit_card import card

"# Streamlit camera input live Demo"
"## Try holding a qr code in front of your webcam"

image = camera_input_live()

if image is not None:
    st.image(image)
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    detector = cv2.QRCodeDetector()

    data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

    if data:
        st.write("# Found QR code")
        st.write(data)
        with st.expander("Show details"):
            st.write("BBox:", bbox)
            st.write("Straight QR code:", straight_qrcode)

res = card(
    title="Streamlit Card",
    text="This is a test card",
    image="/../images/card1.png",
    styles={
        "card": {
            "width": "250px",
            "height": "100px",
            "border-radius": "30px",
            "box-shadow": "0 0 2px rgba(0,0,0,0.5)",
        },
        "filter": {
            "background-color": "rgba(0, 0, 0, 0)"  # <- make the image not dimmed anymore
        },
    },
)

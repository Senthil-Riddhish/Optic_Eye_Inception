from cv2 import log
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
app = FastAPI()
origins = [
    "http://192.168.29.85:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_DIS = tf.keras.models.load_model("../models/1")
MODEL_SC=tf.keras.models.load_model("../models/2")
CLASS_NAMES=['Bulging_Eyes', 'Cataracts', 'Crossed_Eyes', 'Glaucoma', 'Keratoconus', 'Uveitis', 'normal']
CLASS_SCAN=['Cataracts', 'Glaucoma', 'Keratoconus', 'Uveitis', 'normal', 'proptosis']

@app.get("/ping")
async def ping():
    return "hi hello how are you"

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict_disease")
async def predict_disease(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image=tf.image.resize(image,[250,250])
    image = tf.convert_to_tensor(image[:,:,:3])
    img_batch = np.expand_dims(image, 0)
    
    predictions =MODEL_DIS.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print(predicted_class,float(confidence))
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
@app.post("/predict_graph")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image=tf.image.resize(image,[250,250])
    image = tf.convert_to_tensor(image[:,:,:3])
    img_batch = np.expand_dims(image, 0)
    predictions =MODEL_SC.predict(img_batch)
    predicted_class =CLASS_SCAN[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
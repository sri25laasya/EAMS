# dynamic.py
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Annotated
from pydantic import BaseModel
import base64
#import tensorflow
#from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os

from google.cloud import storage
from google.oauth2 import service_account
import cv2
import face_recognition
import os
import numpy as np
import pickle
import datetime

from photos import photos_print



with open('model1.pkl', 'rb') as f:
    known_faces,known_names = pickle.load(f)

app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/main",response_class=HTMLResponse)
def collect_video(request:Request):
    video_path = '/workspace/EAMS/video1.mp4'
    a=photos_print(video_path)
    return templates.TemplateResponse("returning.html", {"request": request, "video_path":video_path})

# Clean up temporary files
def cleanup():
    if os.path.exists("video1.mp4"):
        os.remove("video1.mp4")
    if os.path.exists("/workspace/EAMS/attendance.csv"):
        os.remove("/workspace/EAMS/attendance.csv")

@app.on_event("shutdown")
async def on_shutdown():
    cleanup()

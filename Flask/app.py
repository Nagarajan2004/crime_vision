import re
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from flask import Flask, app,request,render_template
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.models import load_model

#Loading the model
model=load_model(r"crime.h5")

app=Flask(__name__)


#home page 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict.html')
def predict():
    return render_template('predict.html')

@app.route('/index.html')
def home():
    return render_template("index.html")


@app.route('/result',methods=["GET","POST"])
def res():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__) 
        filepath=os.path.join(basepath,'uploads',f.filename) 
        f.save(filepath)


        img = image.load_img(filepath,target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        pred = np.argmax(model.predict(x))
        op = ['Fighting','Arrest','Vandalism','Assault','Stealing','Arson','NormalVideos','Burglary','Explosion','Robbery','Abuse','Shooting','Shoplifting','RoadAccidents'] # Creating list
        op[pred]
        result = op[pred]
        return render_template('predict.html',pred=result)
        



if __name__ == "__main__":
    app.run(debug = True)
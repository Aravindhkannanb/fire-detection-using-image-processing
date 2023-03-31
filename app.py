from flask import Flask,request,render_template
import cv2
from flask_ngrok import run_with_ngrok
import numpy as np
from tensorflow import keras
import pandas as pd
import joblib
def preprocessing_image(filepath):
  img = cv2.imread(filepath) #read
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
  img = cv2.resize(img,(196,196))  # resize
  img = img / 255 #scale
  return img 
app=Flask(__name__)
@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="POST":
        cam=cv2.VideoCapture(0)
        while True:
            ret,frame=cam.read()
            cv2.imshow("Face Image",frame)
            if cv2.waitKey(100)&0xFF==ord("q"):
                break
        cv2.imwrite("Face.png",frame)
        new_predict=keras.models.load_model("F:\\Python\\firedetection\\fire.h5")
        l=[]
        l.append(preprocessing_image("Face.png"))
        y_pred=new_predict.predict(np.array(l))
        y_pred=y_pred.reshape(-1)
        y_pred = y_pred.reshape(-1)
        y_pred[y_pred<0.7] = 0
        y_pred[y_pred>=0.7] = 1
        y_pred = y_pred.astype('int')
        if y_pred[0]==1:
            data="fire"
        else:
            data="non-fire"
        return render_template("home.html",data=data)
    return render_template("home.html")
if __name__=="__main__":
    app.run()
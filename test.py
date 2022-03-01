import pyaudio
import os
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier


from utils import extract_feature

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def upload():
    # if request.method == 'POST':
        
    #     f = request.files['file']

        
    #     basepath = os.path.dirname(__file__)
    #     if not os.path.exists('uploads'):
    #         os.mkdir('uploads')
    #     file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    #     f.save(file_path)

        #Prediction

        # load the saved model (after training)
        try:

            model = pickle.load(open("result/mlp_classifier.model", "rb"))
    # extract features and reshape it
            features = extract_feature("test.wav", mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
            result =  'you are '+ model.predict(features)[0]
    # show the result !
            print("result:", result)
            return render_template('pred.html',emo=result)
        except:
            return render_template('pred.html',emo='No file found')
    # return None


if __name__ == '__main__':
    app.run(debug=True)
    


        
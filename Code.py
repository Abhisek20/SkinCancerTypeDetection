from sys import path
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pathlib



try:
	import shutil
	shutil.rmtree('uploaded/images')
	#% cd uploaded % mkdir image % cd ..
	print()
except:
	pass


directory = str(pathlib.Path(__file__).parent.resolve())
#directory = directory.replace('/', '"\"')
model = tf.keras.models.load_model(directory + '/model.h5')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = directory +'/uploaded/images'

@app.route('/')
def upload_f():
	return render_template('upload.html')

def finds(ImagePath):
	
	class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 
            		'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 
            		'squamous cell carcinoma', 'vascular lesion'] # change this according to what you've trained your model to do

	
	img_height, img_width  = (180, 180)
	
	test_img_ds = tf.keras.preprocessing.image.load_img(ImagePath, 
                        target_size = (img_height, img_width))
	
	img_array = tf.keras.preprocessing.image.img_to_array(test_img_ds)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])
	statement = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
	return statement


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
		f.save(img_path)
		val = finds(img_path)
		return render_template('pred.html', ss = val)

if __name__ == '__main__':
	app.run(debug= False)

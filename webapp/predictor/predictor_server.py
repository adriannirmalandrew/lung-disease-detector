#Flask:
from flask import Flask, render_template, request
import cv2
#Make sure we can load libdarknet.so
import os
os.environ['DARKNET_PATH'] = './darknet/'
#Make sure we can load the python bindings:
import sys
sys.path.insert(0, './darknet')
#Import darknet libs:
import darknet, darknet_images
#Get current UNIX timestamp:
import time

app = Flask(__name__)

@app.route("/predictor", methods = ['POST'])
def predict():
	#Save uploaded xray image:
	xray_file = request.files['xray']
	xray_path = './xray_temp/' + xray_file.filename
	xray_file.save(xray_path)
	#Detect symptoms:
	start_time = time.time()
	network, class_names, class_colors = darknet.load_network('./darknet/test.cfg', './darknet/obj.data', './darknet/tiny-089.weights', 1)
	pred_image, detections = darknet_images.image_detection(xray_path, network, class_names, class_colors, 0.15)
	detect_time = time.time() - start_time
	#Save new image in temp folder:
	pred_name = xray_file.filename + '_pred.jpg'
	cv2.imwrite('../pred_temp/' + pred_name, pred_image)
	#Delete original file:
	os.remove(xray_path)
	#Return template:
	return render_template('predictor.html', name = request.form['name'], sex = request.form['sex'], age = request.form['age'], detect_img = '/pred_temp/' + pred_name, predictions = detections, detect_time = detect_time)

if __name__ == '__main__':
	app.run()
"""
This is how models inference and show up
"""
# 1. Packages - Import
import base64
import io
import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, url_for, flash, redirect, request, jsonify
import cv2
import tensorflow as tf

from utils.yolo_utils import read_classes, read_anchors, yolo_head
from utils.yolo_utils import generate_colors, draw_outputs, yolo_eval
from utils.image_utils import preprocess_image, preprocess_faceimage
from utils.image_utils import img_to_array, triplet_loss, draw
from utils.forms import RegistrationForm, LoginForm
from utils.priorbox import PriorBox
from utils.inception_blocks_v2 import faceRecoModel
from utils.fr_utils import load_weights_from_FaceNet


warnings.filterwarnings("ignore")

# 2. APP - Define
app = Flask(__name__)

# 3. APP - Security
app.config['SECRET_KEY'] = '5007b4292564f59244169bfcbf8eef56'

# 4. Global Const - Define
TEMP_DIR = '../temp/'
MODEL_DIR = '../models/'
model_list = {
    '猫狗图像识别': 'mobilenet_2_dog_cat.h5',
    '物体图像识别': 'mobilenet_1000_imagenet.h5',
    '手势图像识别': 'mobilenet_10_gesture.h5',
    '人脸识别': 'facenet_2_verify.h5',
    '人脸检测': 'ssd_face_detect.onnx',
    '目标检测': 'yolov3_80_object_detect.h5',
    '图像分割': '',
    'NLP-实体识别':''
}
# 5. Utility Functions - Define


def get_model(modelname):
    """
    Get model name and load them
    """
    global MODEL
    if modelname == "facenet_2_verify.h5":
        if not os.path.exists(MODEL_DIR+modelname):
            MODEL = faceRecoModel(input_shape=(96, 96, 3))
            MODEL.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
            load_weights_from_FaceNet(MODEL)
            MODEL.save(MODEL_DIR+modelname)
        MODEL = tf.keras.models.load_model(MODEL_DIR+modelname,
        custom_objects={"triplet_loss": triplet_loss})            

    elif modelname == 'ssd_face_detect.onnx':
        MODEL = cv2.dnn.readNet(MODEL_DIR+modelname)
        MODEL.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        MODEL.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        print(MODEL_DIR+modelname)
        MODEL = tf.keras.models.load_model(MODEL_DIR+modelname)
    print(" * MODEL loaded!")

# 6. Decorators - Define
# 6.1 home page


@app.route('/')
@app.route('/home')
def home():
    """
    Load home page
    """
    return render_template('home.html')

# 6.2 Web service #1 - Load model


@app.route("/loadmodel", methods=["POST"])
def loadmodel():
    """
    Load model for inference
    """
    message = request.get_json(force=True)
    modelname = message['modelename']
    get_model(model_list[modelname])
    response = {
        'loaded': 1
    }
    return jsonify(response)

# 6.3 Web Service #2 - Predict


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main function to make predictions based on all kinds of models
    """
    # Get info from web client and decode info into two fields 'image' and 'model_selected'
    message = request.get_json(force=True)
    encoded = message['image']
    model_selected = message['model_seleted']
    # get bytes IO's like image and turn it to formal image
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))

    # Call 1st model's predict function based on image data
    if model_selected == '猫狗图像识别':
        # preprocess the image
        processed_image = preprocess_image(image, target_size=(224, 224))

        prediction = MODEL.predict(processed_image).tolist()
        # Create response message to web client
        response = {
            'prediction': {
                'dog': prediction[0][1],
                'cat': prediction[0][0]
            }
        }
    # 2nd model
    elif model_selected == '物体图像识别':
        # preprocess the image
        processed_image = preprocess_image(image, target_size=(224, 224))

        # further process to satify model requirement
        prediction = MODEL.predict(
            tf.keras.applications.mobilenet.preprocess_input(processed_image))
        # Call imagenet util to decode preditions
        top5 = tf.keras.applications.imagenet_utils.decode_predictions(
            prediction)
        # Turn the result into dataframe to ease handle
        top5_df = pd.DataFrame(data=top5[0], columns=[
                               'id', 'name', 'possibility'])
        top5_df['possibility'] = top5_df['possibility'].astype(np.float)
        for i in range(len(top5_df)):
            print("Top%d is %s with possibility(%.2f%%)." %
                  (i+1, top5_df['name'][i], top5_df['possibility'][i]*100))
        # Create response message to web client
        top5_dict = top5_df.to_dict()
        response = {
            'prediction': top5_dict
        }
    elif model_selected == '手势图像识别':
        # preprocess the image
        processed_image = preprocess_image(image, target_size=(224, 224))

        prediction = MODEL.predict(
            tf.keras.applications.mobilenet.preprocess_input(processed_image))
        rounded_prediction = np.argmax(prediction, axis=1)
        response = {
            'prediction': {
                'number': np.float(rounded_prediction[0])
            }
        }
    elif model_selected == '人脸识别':
        # preprocess the 1st image
        p_faceimage = preprocess_faceimage(image, target_size=(96, 96))
        encoding_db = MODEL.predict(p_faceimage)

        # process 2nd image
        encoded_2 = message['image_2']
        # get bytes IO's like image and turn it to formal image
        decoded_2 = base64.b64decode(encoded_2)
        image_2 = Image.open(io.BytesIO(decoded_2))
        p_faceimage_2 = preprocess_faceimage(image_2, target_size=(96, 96))
        encoding_capture = MODEL.predict(p_faceimage_2)

        dist = np.linalg.norm(encoding_capture-encoding_db)
        print(dist)
        # Create response message to web client
        response = {
            'prediction': {
                'distance': np.float(dist)
            }
        }
    elif model_selected == '人脸检测':
        # get height & width of original image
        #_, h, w, _ = image.shape
        target_h, target_w = 416, 416
        conf_thresh = 0.6
        nms_thresh = 0.3
        keep_top_k = 5

        processed_image = image.resize((target_w, target_h))
        processed_image = img_to_array(processed_image)

        processed_image_normalized = cv2.dnn.blobFromImage(processed_image)
        print("processed_image_normalized's shape =",processed_image_normalized.shape)

        # 模型预测
        output_names = ['loc', 'conf', 'iou']
        MODEL.setInput(processed_image_normalized)
        loc, conf, iou = MODEL.forward(output_names)

        # Decode bboxes and landmarks
        prior_box = PriorBox(input_shape=(target_w, target_h),
                      output_shape=(target_w, target_h))
        dets = prior_box.decode(loc, conf, iou, conf_thresh)

        # NMS
        img_cut = []
        if dets.shape[0] > 0:
            # NMS from OpenCV
            keep_idx = cv2.dnn.NMSBoxes(
                bboxes=dets[:, 0:4].tolist(),
                scores=dets[:, -1].tolist(),
                score_threshold=conf_thresh,
                nms_threshold=nms_thresh,
                eta=1,
                top_k=keep_top_k
            )  # returns [box_num, box_detailed_info]

            selected_dets = dets[keep_idx]
            print(f'Detected {selected_dets.shape[0]} faces successfully!')
            for det in selected_dets:
                print(f'Central point[X,Y]=[{det[0]:.1f}, {det[1]:.1f}] Width,Height=[{det[2]:.1f},\
                    {det[3]:.1f}] Confidence Score={det[-1]:.2f}')

                margin = 30
                x_left = np.uint16(det[0])-margin
                x_right = np.uint16(det[0]+det[2])+margin
                y_top = np.uint16(det[1])-margin
                y_bottom = np.uint16(det[1]+det[3])+margin
                print(f'Topleft point[X,Y]=[{x_left}, {y_top}] \
                    Bottomright point[X,Y]=[{x_right}, {y_bottom}]')

                img_cut.append(processed_image[y_top:y_bottom, x_left:x_right])
        else:
            print('No faces found.')
            exit()

        # Draw boudning boxes and landmarks on the original image
        img_res = draw(
            img=processed_image,
            bboxes=selected_dets[:, :4],
            landmarks=np.reshape(selected_dets[:, 4:14], (-1, 5, 2)),
            scores=selected_dets[:, -1]
        )

        img_file = Image.fromarray(np.uint8(img_res))
        img_filename = 'images/face_with_box.jpg'
        img_file.save(img_filename)
 
        with open(img_filename, 'rb') as img_f:
            img_stream = img_f.read()
            decoded = base64.b64encode(img_stream).decode()
            response = {
                'prediction': {
                    'image': decoded
                }
            }
    elif model_selected == '目标检测':
        # preprocess the image
        processed_image = preprocess_image(image, target_size=(416, 416))
        processed_image_normalized = np.around(
            processed_image/255.0, decimals=12)
        # read yolo model data
        class_names = read_classes(MODEL_DIR+"model_data/coco_classes.txt")
        anchors = read_anchors(MODEL_DIR+"model_data/yolo_anchors.txt")
        # get model outputs
        yolo_outputs = MODEL(processed_image_normalized)
        # post-process outputs
        outputs = yolo_head(yolo_outputs, anchors, len(class_names))
        # 过滤边框
        out_scores, out_boxes, out_classes = yolo_eval(outputs)
        # 产生边框的颜色
        colors = generate_colors(class_names)
        print(f'Found {len(out_boxes)} boxes for image')
        #print('Found {} boxes for image'.format(len(out_boxes)))
        img_squeezed = np.squeeze(processed_image, axis=0)
        img = draw_outputs(img_squeezed, out_scores, out_boxes,
                           out_classes, colors, class_names)
        # turn to img object from np.array and save to a image file to check it
        img_file = Image.fromarray(np.uint8(img))
        img_filename = TEMP_DIR+'face_with_box.jpg'
        img_file.save(img_filename)
        print("saved")
        with open(img_filename, 'rb') as img_f:
            img_stream = img_f.read()
            decoded = base64.b64encode(img_stream).decode()
            response = {
                'prediction': {
                    'image': decoded
                }
            }
    else:
        print("No recognized model")
    return jsonify(response)


@app.route('/about')
def about():
    """
    'About' page - to be constructed
    """
    return render_template('about.html', title='About')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    'Register' page - to be constructed
    """
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    'Login' page - to be constructed
    """
    form = LoginForm()
    return render_template('login.html', title='Login', form=form)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)

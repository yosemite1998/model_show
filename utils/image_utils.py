import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

def preprocess_image(image, target_size):
    """
    Use pillow to pre-precess the image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_faceimage(image, target_size):
    """
    Use pillow to pre-precess the face image
    """
    image = preprocess_image(image, target_size)
    image = np.around(image/255.0, decimals=12)
    return image

# 实现三元组损失函数
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    参数:
    y_true -- 这个参数暂时不用
    y_pred -- 这是一个python列表，里面包含了3个对象,分别是A,P,N的编码。
    返回值:
    loss -- 损失值
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # 计算A与P的编码差异，就是公式中标（1）的那块
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # 计算A与N的编码差异，就是公式中标（2）的那块
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # 根据两组编码差异来计算损失
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)
    
    return loss    

from typing import List, Tuple

def draw(img: np.ndarray, bboxes: np.ndarray, landmarks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    '''
    This function draws bounding boxes and landmarks on the image and return the result.

    Parameters:
        bboxes    - bboxes of shape [n, 5]. 'n' for number of bboxes, '5' for coordinate and confidence
                    (x1, y1, x2, y2, c).
        landmarks - landmarks of shape [n, 5, 2]. 'n' for number of bboxes, '5' for 5 landmarks
                    (two for eyes center, one for nose tip, two for mouth corners), '2' for coordinate
                    on the image.
    Returns:
        img       - image with bounding boxes and landmarks drawn
    '''

    # draw bounding boxes
    if bboxes is not None:
        color = (0, 255, 0)
        thickness = 2
        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx].astype(np.int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, thickness)
            cv2.putText(img, '{:.4f}'.format(scores[idx]), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # draw landmarks
    if landmarks is not None:
        radius = 2
        thickness = 2
        color = [
            (255,   0,   0), # right eye
            (  0,   0, 255), # left eye
            (  0, 255,   0), # nose tip
            (255,   0, 255), # mouth right
            (  0, 255, 255)  # mouth left
        ]
        for idx in range(landmarks.shape[0]):
            face_landmarks = landmarks[idx].astype(np.int)
            for idx, landmark in enumerate(face_landmarks):
                cv2.circle(img, (int(landmark[0]), int(landmark[1])), radius, color[idx], thickness)
    return img

import tensorflow as tf
import numpy as np
import cv2
import colorsys
import random
from IPython.display import display

# As tensorflow lite doesn't support tf.size used in tf.meshgrid, 
# we reimplemented a simple meshgrid function that use basic tf function.
def _meshgrid(n_a, n_b):

    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]


def yolo_head(preds, anchors, classes): #preds中有3个pred
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    outputs = {}
    
    for i in range(3):
        pred = preds[i]
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        #pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = _meshgrid(grid_size[1],grid_size[0])
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
            tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors[[6-i*3,7-i*3,8-i*3]]

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        outputs['output'+str(i)] = (objectness, bbox, class_probs)

    return (outputs['output0'],outputs['output1'],outputs['output2'])

# 过滤掉那些概率低的边框
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """
    box_confidence -- 装载着每个边框的pc
    boxes -- 装载着每个边框的坐标
    box_class_probs -- 装载着每个边框的80个种类的概率
    threshold -- 阈值，概率低于这个值的边框会被过滤掉
    
    返回值:
    scores -- 装载保留下的那些边框的概率
    boxes -- 装载保留下的那些边框的坐标
    classes -- 装载保留下的那些边框的种类的索引
    
    """
    
    # 将pc和c相乘，得到具体某个种类是否存在的概率
    box_scores = box_confidence * box_class_probs
    
    
    box_classes = tf.argmax(box_scores, axis=-1) # 获取概率最大的那个种类的索引
    box_class_scores = tf.reduce_max(box_scores, axis=-1) # 获取概率最大的那个种类的概率值

    
    # 创建一个过滤器。当某个种类的概率值大于等于阈值threshold时，
    # 对应于这个种类的filtering_mask中的位置就是true，否则就是false。
    # 所以filtering_mask就是[False, True, 。。。, False, True]这种形式。
    filtering_mask = tf.greater_equal(box_class_scores, threshold)

    
    # 用上面的过滤器来过滤掉那些概率小的边框。
    # 过滤完成后，scores和boxes，classes里面就只装载了概率大的边框的概率值和坐标以及种类索引了。
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

# 用非最大值抑制技术过滤掉重叠的边框    
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 20, iou_threshold = 0.5):
    """
    参数:
    scores -- 前面yolo_filter_boxes函数保留下的那些边框的概率值
    boxes -- 前面yolo_filter_boxes函数保留下的那些边框的坐标
    classes -- 前面yolo_filter_boxes函数保留下的那些边框的种类的索引
    max_boxes -- 最多想要保留多少个边框
    iou_threshold -- 交并比，这是一个阈值，也就是说交并比大于这个阈值的边框才会被进行非最大值抑制处理
    
    Returns:
    scores -- NMS保留下的那些边框的概率值
    boxes -- NMS保留下的那些边框的坐标
    classes -- NMS保留下的那些边框的种类的索引
    """
    
    # tensorflow为我们提供了一个NMS函数，我们直接调用就可以了tf.image.non_max_suppression()。
    # 这个函数会返回NMS后保留下来的边框的索引
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=iou_threshold)

    # 通过上面的索引来分别获取被保留的边框的相关概率值，坐标以及种类的索引
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes

# 这个函数里面整合了前面我们实现的两个过滤函数。将YOLO模型的输出结果输入到这个函数后，这个函数会将多余边框过滤掉
def yolo_eval(outputs, max_boxes=20, score_threshold=.5, iou_threshold=.5):
    """
    参数:
    yolo_outputs -- YOLO模型的输出结果
    max_boxes -- 你希望最多识别出多少个边框
    score_threshold -- 概率值阈值
    iou_threshold -- 交并比阈值
    
    Returns:
    scores -- 最终保留下的那些边框的概率值
    boxes -- 最终保留下的那些边框的坐标
    classes -- 最终保留下的那些边框的种类的索引
    """
    #建立3个空list
    s, b, c = [], [], []
    
    # 我们后面我们调用的Yolov3使用了3个规格的网格（13*13，26*26，52*52）进行预测。所以有3组output。
    for output in outputs:
        
        ### 将YOLO输出结果分成3份，分别表示概率值，坐标，种类索引
        box_confidence, boxes,  box_class_probs = output

        # 使用我们前面实现的yolo_filter_boxes函数过滤掉概率值低于阈值的边框
        scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, 
                                                   threshold = score_threshold)
        
        s.append(scores)
        b.append(boxes)
        c.append(classes)
    
    #将3组output的结果整合到一起
    scores = tf.concat(s, axis=0)
    boxes = tf.concat(b, axis=0)
    classes = tf.concat(c, axis=0)
    

    # 使用我们前面实现的yolo_non_max_suppression过滤掉重叠的边框
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, 
                                                      iou_threshold = iou_threshold)
      
    return scores, boxes, classes

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split()]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10201)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


def preprocess_image_yolo(img_path, model_image_size):
    img_raw = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = tf.image.resize(img, model_image_size)/255.
    return img_raw, img

def draw_outputs(img, out_scores, out_boxes, out_classes, colors, class_names):

    wh = np.flip(img.shape[0:2])
    for i,c in list(enumerate(out_classes)):
        x1y1 = tuple((np.array(out_boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(out_boxes[i][2:4]) * wh).astype(np.int32))
        x1y1_lable = tuple((np.array(out_boxes[i][0:2]) * wh + [0,-15]).astype(np.int32))
        x2y2_lable = tuple((np.array(out_boxes[i][0:2]) * wh + [(len(class_names[int(out_classes[i])])+6)*12,0]).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, colors[c], 2)
        img = cv2.rectangle(img, x1y1_lable, x2y2_lable, colors[c], -1)
        img = cv2.putText(img, '{} {:.2f}'.format(
            class_names[int(out_classes[i])], out_scores[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        print('{} {:.2f}'.format(class_names[int(out_classes[i])], out_scores[i]),
              x1y1, x2y2)
    return img


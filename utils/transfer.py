import tensorflow as tf

model = tf.keras.models.load_model('mobile_dog_cat.h5')
model.save("mobilenet_dot_cat")

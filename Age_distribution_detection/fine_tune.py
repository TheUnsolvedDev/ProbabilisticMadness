import tensorflow as tf

from model import model_cnn

if __name__ == '__main__':
    model_keras = model_cnn()
    model_keras.load_weights('modelA*.h5')
    model_keras.save('model_keras.h5')
    model_keras= tf.keras.models.load_model('model_keras.h5')
    model = model_keras
    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    tflite_model = converter.convert()
    
    export_dir = 'saved_model/1'
    tf.saved_model.save(model, export_dir)
    # Converting a SavedModel to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()
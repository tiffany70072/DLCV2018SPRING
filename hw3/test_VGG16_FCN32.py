import sys
from train_FCN16 import VGG16_FCN16
from train_FCN32 import VGG16_FCN32
from visualization import test_import_images, test_output_one
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)

test_dir = sys.argv[1]
output_dir = sys.argv[2]
sat, idx = test_import_images(test_dir)
print("Load model...")
model = VGG16_FCN32('hw3_model/VGG16FCN32_epoch40_weights.h5')
#model.summary()
print("Predict and save images...")
for i in range(sat.shape[0]):
	pred = model.predict(sat[i:i+1])
	test_output_one(pred[0], idx[i], output_dir)

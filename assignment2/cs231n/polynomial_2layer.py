import tensorflow as tf
import numpy as np


BATCH_SIZE = 200
NUM_TRAIN = 800
NUM_VAL = 100
NUM_TEST = 100
TRAIN_STEPS = 10000

dtype = tf.float32


# Create datas
x_data = 10*(np.random.random(1000) - 0.5).reshape(1000,1)
y_data = 0.1*x_data**2 - 6*x_data - 0.1*x_data**3

x_data = np.concatenate((x_data, x_data**2, x_data**3), axis=1)

x_data -= np.mean(x_data)
x_data /= np.std(x_data, axis=0)
print('x_data shape : ' + str(x_data.shape))
print('x_data :')
print(x_data[:20])
#5*x_data**5 - 9*x_data**4 +
X_train, y_train = x_data[:NUM_TRAIN], y_data[:NUM_TRAIN]
X_val, y_val = x_data[NUM_TRAIN:NUM_TRAIN+NUM_VAL], y_data[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
X_test, y_test = x_data[-NUM_TEST:], y_data[-NUM_TEST:] 
print('x_train shape : ' + str(X_train.shape))


# Create layers
def addlayer(name, input_tensor, output_size, activition_function, l2_reg=0):
	input_size = int(input_tensor.shape[1])
	regularizer = tf.contrib.layers.l2_regularizer(l2_reg) if not l2_reg == 0  else None
	W = tf.get_variable(name+'_W', [input_size, output_size], dtype, tf.contrib.layers.xavier_initializer(),
		regularizer=regularizer)
	b = tf.get_variable(name+'_b', [output_size], dtype, tf.zeros_initializer(),
		regularizer=regularizer)
	h = tf.nn.relu(tf.add(tf.matmul(input_tensor, W), b))
	
	return h
def get_loss(y_pred, y, use_reg):
	loss = tf.reduce_mean(tf.square(y_pred - y))
	if use_reg:
		regularization_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		loss += regularization_loss
	return loss

graph = tf.Graph()
with graph.as_default():
	X = tf.placeholder(dtype, [None, x_data.shape[-1]])
	y = tf.placeholder(dtype, [None,1])
	h1 = addlayer('layer1', X, 100, tf.nn.relu)
	h2 = addlayer('layer2', h1, 1000, tf.nn.relu)
	o = addlayer('layer3', h2, 100, tf.nn.relu)


	"""layer2"""
	W2 = tf.get_variable('W2', [100,1], dtype, tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable('b2', [1], dtype, tf.zeros_initializer())
	y_ = tf.matmul(o, W2) + b2

	loss = get_loss(y_, y, use_reg=False)
	train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session(graph=graph) as sess:
	sess.run(tf.global_variables_initializer())
	num_train = X_train.shape[0]
	for e in range(epochs):
		for i in range(num_train//BATCH_SIZE):
			mask = np.random.choice(num_train, BATCH_SIZE)
			x_train_batch = X_train[mask]
			y_train_batch = y_train[mask]
			_, train_loss = sess.run([train_step, loss], feed_dict={X:x_train_batch, 
					y:y_train_batch.reshape((-1,1))})
			if i % 100 == 0 or i+1==TRAIN_STEPS:
				print('Step: %d, Loss: %.5f' %(i, train_loss))
	val_loss = sess.run(loss, feed_dict={X:X_val, y: y_val.reshape(-1,1)})
	print('Validation Loss: %.3f' %val_loss)





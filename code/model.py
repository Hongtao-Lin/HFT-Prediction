#!/usr/bin/python
# -*- coding:utf8 -*-

import json, re, os, sys, timeit, time, gzip
import h5py, cPickle
import numpy as np
import theano
import theano.tensor as T
from theano.compile.debugmode import DebugMode

def load_data(dataset):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''

	#############
	# LOAD DATA #
	#############

	# Download the MNIST dataset if it is not present
	data_dir, data_file = os.path.split(dataset)
	if data_dir == "" and not os.path.isfile(dataset):
		# Check if dataset is in the data directory.
		new_path = os.path.join(
			os.path.split(__file__)[0],
			"..",
			"data",
			dataset
		)
		if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
			dataset = new_path

	if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
		import urllib
		origin = (
			'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		)
		print 'Downloading data from %s' % origin
		urllib.urlretrieve(origin, dataset)

	print '... loading data'

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#witch row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	return train_set, valid_set, test_set


def load_split_data(fname):
	f = h5py.File(fname, "r")
	x = f["data"][:]
	y = f["label"][:]
	f.close()
	print y
	data_size = y.shape[0]

	idx = np.random.permutation(data_size)
	train_idx = idx[:0.8*data_size]
	dev_idx = idx[0.8*data_size:0.9*data_size]
	test_idx = idx[0.9*data_size:]

	train = [x[train_idx], y[train_idx]]
	dev = [x[dev_idx], y[dev_idx]]
	test = [x[test_idx], y[test_idx]]

	return train, dev, test

def _init_hidden_weights(n_in, n_out):
	rng = np.random.RandomState(1234)
	weights = np.asarray(
		rng.uniform(
			low=-np.sqrt(6. / (n_in + n_out)),
			high=np.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)
		),
		dtype=theano.config.floatX
	)
	bias = np.zeros((n_out,), dtype=theano.config.floatX)
	return (
		theano.shared(value=weights, name='W_h', borrow=True),
		theano.shared(value=bias, name='b_h', borrow=True)
	)

def _init_logreg_weights(n_in, n_out):
	rng = np.random.RandomState(1234)
	weights = np.asarray(
		rng.normal(
			0, 0.1, (n_in, n_out)
		),
		dtype=theano.config.floatX
	)
	weights = np.zeros((n_in, n_out), dtype=theano.config.floatX)
	bias = np.zeros((n_out,), dtype=theano.config.floatX)
	return (
		theano.shared(name='W', borrow=True, value=weights),
		theano.shared(name='b', borrow=True, value=bias)
	)

def feed_forward(activation, weights, bias, input):
	return activation(T.dot(input, weights) + bias)

def sgd(param, cost, lr):
	return param - (lr * T.grad(cost, param))

def L1(reg, w1, w2):
	return reg * (abs(w1).sum() + abs(w2).sum())

def L2(reg, w1, w2):
	return reg * ((w1 ** 2).sum() + (w2 ** 2).sum())

def build_model(n_in, n_classes, n_hidden, lr, L1_reg, L2_reg):
	# add comments...

	# Q: why use different types?? 
	# Accroding to inherit properties of x, y.
	x = T.matrix('x')
	y = T.lvector('y')

	W_h, b_h = _init_hidden_weights(n_in, n_hidden)
	W, b = _init_logreg_weights(n_hidden, n_classes)

	p_y_given_x = feed_forward(T.nnet.softmax, 
					W, b,
					feed_forward(
						T.tanh, W_h, b_h, x
				  ))


	cost2 = -T.sum(T.log(p_y_given_x[:,y])) + L1(L1_reg, W, b) + L2(L2_reg, W, b)
	cost = -T.sum(p_y_given_x[:,y]) + L1(L1_reg, W, b) + L2(L2_reg, W, b)
	acc = T.sum(T.neq(y, T.argmax(p_y_given_x, axis=1))) / (y.shape[0]*1.)
	# debug = theano.printing.Print("debug")
	train_model = theano.function(
		inputs = [x, y],
		# outputs = [cost2, debug(acc)],
		outputs = cost2,
		updates = [
			(W, sgd(W, cost, lr)),
			(b, sgd(b, cost, lr)),
			(W_h, sgd(W_h, cost, lr)),
			(b_h, sgd(b_h, cost, lr)),
			]
		)

	
	evaluate_model = theano.function(
		inputs = [x, y],
		outputs = acc
		)

	return train_model, evaluate_model

def main(lr = 1e-5, L1_reg = 0, L2_reg = 1e-4, n_epochs = 10, 
		 n_hidden = 500, dataset="data/data.hdf5"):
	# train, dev, test = load_data("data/mnist.pkl.gz")
	train, dev, test = load_split_data(dataset)
	n_in = train[0].shape[1]
	n_classes = max(train[1]) - min(train[1]) + 1
	print n_in, n_classes
	print "building model..."
	train_model, evaluate_model = build_model(n_in, n_classes, n_hidden, lr, L1_reg, L2_reg)
	print "building model complete..."

	# Add early stoping!
	print "training..."
	batch_size = 50
	n_batches = train[0].shape[0] // batch_size
	patience = 10000
	patience_inc = 2
	improv_threshold = 0.995
	validate_freq = min(n_batches, patience//2)

	best_valid_loss = np.inf
	best_iter = 0
	test_score = 0.

	epoch = 0
	done_loop = False

	while (epoch < n_epochs) and (not done_loop):
		epoch += 1
		for batch_idx in range(n_batches):
			batch_x = train[0][batch_idx*batch_size: (batch_idx+1)*batch_size]
			batch_y = train[1][batch_idx*batch_size: (batch_idx+1)*batch_size]
			cost = train_model(batch_x, batch_y)
			# print cost
			# time.sleep(0.5)
			iter = (epoch-1)*n_batches + batch_idx
			if (iter + 1) % validate_freq == 0:
				valid_loss = evaluate_model(dev[0], dev[1])

				print('epoch %i, minibatch %i/%i, validation error %f %%' %(
						epoch,
						batch_idx + 1,
						n_batches,
						valid_loss * 100.))

				if valid_loss < best_valid_loss:
					if (valid_loss < best_valid_loss * improv_threshold):
						patience = max(patience, iter*patience_inc)
					best_valid_loss = valid_loss
					best_iter = iter
					test_loss = evaluate_model(test[0], test[1])
					print(('     epoch %i, minibatch %i/%i, test error of '
						   'best model %f %%') %
						  (epoch, batch_idx + 1, n_batches,
						   test_score * 100.))
			if patience <= iter:
				done_loop = True

	print(('Best validation score of %f %% '
		   'obtained at iteration %i, with test performance %f %%') %
		  (best_valid_loss * 100., best_iter + 1, test_score * 100.))

if __name__ == '__main__':
	main()
	pass

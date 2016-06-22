import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import math, random
from sklearn import preprocessing

def read_data(fname1, fname2):
	f = h5py.File(fname1, "r")
	actual_price = f["price"][:]
	f.close()
	f = h5py.File(fname2, "r")
	action = f["label"][:]
	f.close()
	print actual_price.shape
	data_size = action.shape[0]
	actual_price = actual_price[0.9*data_size:]
	action = action[0.9*data_size:]
	return actual_price, action

def evaluate(actual_price, action):
	profit = 0
	rand_profit = 0
	max_profit = 0
	h_profit = 0
	h_act = 1
	for i in range(len(action)-1):
		act = action[i]
		# if actual_price[i] == 0 or actual_price[i+1] == 0:
		# 	continue
		if act == 1:
			profit += (actual_price[i+1] - actual_price[i])
		else:
			profit += (actual_price[i] - actual_price[i+1])
		max_profit += math.fabs(actual_price[i] - actual_price[i+1])
		h_act = (actual_price[i] < actual_price[i-1])
		if h_act == 1:
			h_profit += (actual_price[i+1] - actual_price[i])
		else:
			h_profit += (actual_price[i] - actual_price[i+1])
		print act, profit
	iter_num = 100
	for it in range(iter_num):
		for i in range(len(action)-1):
			if actual_price[i] == 0 or actual_price[i+1] == 0:
				continue
			r_act = random.randint(0, 1)
			if r_act == 1:
				rand_profit += (actual_price[i+1] - actual_price[i])
			else:
				rand_profit += (actual_price[i] - actual_price[i+1])
	rand_profit = rand_profit / iter_num

	print len(actual_price) 
	print "rand_profit is: ", rand_profit
	print "h_profit is: ", h_profit
	print "profit is: ", profit
	print "max_profit is: ", max_profit

def draw_stat(actual_price, action):
	price_list = []
	x_list = []
	# idx = np.where(actual_price == 0)[0]
	# print idx
	# print actual_price[np.where(actual_price < 2000)]
	# idx = [0] + idx.tolist()
	# print idx
	# for i in range(len(idx)-1):
	# 	price_list.append(actual_price[idx[i]+1:idx[i+1]-1])
	# 	x_list.append(range(idx[i]+i+1, idx[i+1]+i-1))
	# for i in range(len(idx)-1):
	# 	print x_list[i]
	# 	print price_list[i]
	# 	plt.plot(x_list[i], price_list[i], 'r')
	x_list = range(1,50)
	price_list = actual_price[1:50]
	plt.plot(x_list, price_list, 'k')
	for i in range(1, 50):
		style = 'go'
		if action[i] == 1:
			style = 'ro'
		plt.plot(i, actual_price[i], style)
	plt.ylim(2140, 2144.2)
	# plt.show()
	plt.savefig("action.png")

if __name__ == '__main__':
	fname1 = "data/IH1605.hdf5"
	fname2 = "data/IH1605_gru_res.hdf5"
	actual_price, action = read_data(fname1, fname2)
	evaluate(actual_price[1:50], action[1:50])
	draw_stat(actual_price, action)
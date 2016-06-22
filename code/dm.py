import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import math
from sklearn import preprocessing

SPINE_COLOR='gray'
# import apollocaffe
# from apollocaffe.layers import 

fnames = ['IC1604', 'IC1605', 'IC1606', 'IC1609', 'IF1604', \
		  'IF1605', 'IF1606', 'IF1609', 'IH1604', 'IH1605', \
		  'IH1606', 'IH1609', 'rb1605', 'rb1610', 'rb1701']
frame = 10

selected_fnames = ['IC1604', 'IH1605']

def read_data(fname):
	df = pd.read_csv(fname)
	return df

def data2vector(df, mode="naive"):
	data = []
	label = []
	price = []

	tradingStart = df["TradingDay"][0]
	tradingEnd = df["TradingDay"][len(df) - 1]	
	for day in range(tradingStart,tradingEnd+1):
		print day
		day_df = df.loc[df["TradingDay"] == day]
		if len(day_df) == 0:
			continue
		
		half_day_df = day_df.loc[day_df["UpdateTime"] < "13:00:00"]
		# Filter out those first / last days
		half_day_df = half_day_df.iloc[20:-20]
		if len(half_day_df) == 0:
			continue

		if mode == "naive":
			x, y, p = df2vector(half_day_df)
		else:
			x, y, p = df2vector_LSTM(half_day_df)
		data += x
		label += y
		price += p

		half_day_df = day_df.loc[day_df["UpdateTime"] > "13:00:00"]
		# Filter out those first / last days
		half_day_df = half_day_df.iloc[20:-20]
		if len(half_day_df) == 0:
			continue

		if mode == "naive":
			x, y, p = df2vector(half_day_df)
		else:
			x, y, p = df2vector_LSTM(half_day_df)
		data += x
		label += y
		price += p

	return np.asarray(data), np.asarray(label), np.asarray(price)

def df2vector(df):
	df["AskPriceMean"]  = (df["AskPrice1"] + df["AskPrice2"] +  df["AskPrice3"] +  df["AskPrice4"] +  df["AskPrice5"]) /5.
	df["BidPriceMean"]  = (df["BidPrice1"] + df["BidPrice2"] +  df["BidPrice3"] +  df["BidPrice4"] +  df["BidPrice5"]) /5.

	# trading End is used to test
	data = []
	label = []
	price = []

	# prepare training data
	# every 60
	# dimension reduction
	norm = 1000
	sub_df = df # slice the day
	# from the first.
	total_length = len(df)
	sub_df.index = np.arange(total_length)
	# print sub_df["LastPrice"][0]
	log_ask_price = np.diff(np.log(sub_df["AskPriceMean"]))
	log_bid_price = np.diff(np.log(sub_df["BidPriceMean"]))

	last_price_mean = np.mean(sub_df["LastPrice"][1:frame+1])
	for i in range(frame*2,total_length-frame, frame):
		# print i
		ask_vector = log_ask_price[i-frame:i].reshape(1, -1)
		bid_vector = log_bid_price[i-frame:i].reshape(1, -1)
		ask_vector = preprocessing.normalize(ask_vector).tolist()
		bid_vector = preprocessing.normalize(bid_vector).tolist()
		fv = ask_vector[0]
		fv += bid_vector[0]
		# fv = np.hstack([fv, np.array(bid_vector)])
		# price mean + price + 
		smooth = []
		this_price_mean = np.mean(sub_df["LastPrice"][i-frame+1:i+1])

		v1 = float((np.log(this_price_mean)-np.log(last_price_mean))*norm)
		v2 = float((np.log(sub_df["LastPrice"][i]) - np.log(last_price_mean))*norm)

		last_price_mean = this_price_mean
		v3 = float(np.var(sub_df['LastPrice'][i-frame+1:i+1]))
		v4 = float(sub_df["Volume"][i] - sub_df["Volume"][i-frame])
		smooth = [v1, v2, v3, v4]
		fv += smooth
		# 124 dimension

		y = int(np.log(sub_df["LastPrice"][i + frame]) - np.log(sub_df["LastPrice"][i]) > 0)
		data.append(fv)
		label.append(y)
		p = sub_df["LastPrice"][i]
		if i == frame*2 or i+frame > total_length-frame:
			p = 0
		price.append(p)


	return data, label, price

def df2vector_LSTM(df):
	# trading End is used to test
	data = []
	label = []
	price = []

	# prepare training data
	# every 60
	# dimension reduction
	norm = 1000
	sub_df = df # slice the day
	# from the first.
	total_length = len(df)
	sub_df.index = np.arange(total_length)
	# print sub_df["LastPrice"][0]
	ask_mean = 0
	bid_mean = 0
	ask_price = []
	bid_price = []
	for f in range(1,6):
		ask_mean += sub_df["AskPrice"+str(f)]
		bid_mean += sub_df["BidPrice"+str(f)]
	ask_mean /= 5
	bid_mean /= 5
	ask_bid_mean = (sub_df["AskPrice1"]+sub_df["BidPrice1"])/2
	for f in range(1,6):
		sub_df["AskPrice"+str(f)] -= ask_mean
		sub_df["BidPrice"+str(f)] -= bid_mean
		ask_price.append(np.asarray(sub_df["AskPrice"+str(f)]))
		bid_price.append(np.asarray(sub_df["BidPrice"+str(f)]))
	for i in range(frame*2,total_length-frame, frame):
		fv = []
		for t in range(i, i+frame):
			v0 = []
			ask_vector = []
			ask_vol = []
			bid_vector = []
			bid_vol = []

			for f in range(1,6):
				ask_vector.append(sub_df["AskPrice"+str(f)][t])
				ask_vol.append(sub_df["AskVolume"+str(f)][t])
				bid_vector.append(sub_df["BidPrice"+str(f)][t])
				bid_vol.append(sub_df["BidVolume"+str(f)][t])

			v0 += ask_vector
			v0 += bid_vector
			v0 += ask_vol
			v0 += bid_vol

			v0.append(sub_df["LastVolume"][t])
			v0.append(sub_df["LastPrice"][t] - ask_bid_mean[t])
			fv.append(v0)

		y = int(np.log(sub_df["LastPrice"][i + frame]) - np.log(sub_df["LastPrice"][i]) > 0)
		data.append(fv)
		label.append(y)
		p = sub_df["LastPrice"][i]
		if i == frame*2 or i+frame > total_length-frame:
			p = 0
		price.append(p)


	return data, label, price


def dump_data(data, label, price, fname):
	f = h5py.File("data/"+fname+".hdf5", "w")

	f["data"] = data
	f["label"] = label
	f["price"] = price

	f.close()


def draw_price(df):
	price_df = df["LastPrice"]
	price_df.plot()
	plt.savefig("IC1604.png")

def draw_trend():
	color_index = ['b', 'g', 'r']
	data = []
	for i in range(len(up)):
		d = [down[i], fair[i], up[i]]
		data.append(d)
	data = np.asarray(data).transpose()
	index = np.arange(data.shape[1])
	print data
	for i in range(data.shape[0]):
		plt.bar(index + .25, data[i], width = .5, color = color_index[i],\
			bottom = np.sum(data[:i], axis = 0), alpha = .7)
	plt.savefig("distribution.png")

def draw_freq():
	color_index = ['b', 'r']
	data = []
	for i in range(len(up)):
		d = [freq[i]]
		data.append(d)
	data = np.asarray(data).transpose()
	index = np.arange(data.shape[1])
	print data
	for i in range(data.shape[0]):
		plt.bar(index + i*.25 + .1, data[i], width = .25, color = color_index[i],\
			alpha = .5)
	plt.savefig("freq.png")

def draw_evaluate():
	xlabel = ["Rand", "Heuristi", "LR", "SVM", "MLP", "RNN", "GRU", "LSTM", "MAX"]
	x = range(9)
	plt.xticks(x, xlabel)
	plt.margins(0.2)
	y1 = [0.1, 18.8, 34.6, 36.1, 36.2, 49.4, 68.1, 63.7, 100]
	y2 = [-0.1, 14.3, 15.7, 15.7, 21.5, 28.0, 59.2, 53.2, 100]
	plt.xlim(-.2, 8.2)
	plt.ylim(-.5, 100)
	plt.plot(x, y1, 'ro-')
	plt.plot(x, y2, 'bo-')
	plt.subplots_adjust(bottom=0.15)
	plt.show()

up = []
down = []
fair = []
freq = []
vol = []
def get_stat(df):
	tradingStart = df["TradingDay"][0]
	tradingEnd = df["TradingDay"][len(df) - 1]
	total_df = 0
	total_vol = 0
	total_cross = 0
	total_up = 0
	total_down = 0
	max_gap_df = 0
	max_gap = 0
	for day in range(tradingStart,tradingEnd+1):
		day_df = df.loc[df["TradingDay"] == day]
		if len(day_df) == 0:
			continue
		half_day_df = day_df.loc[day_df["UpdateTime"] < "13:00:00"]
		# Filter out those first / last days
		half_day_df = half_day_df.iloc[50:-50]

		if len(half_day_df) == 0:
			continue

		cross_df = half_day_df.loc[half_day_df["LastVolume"] > 0]
		sub_df = np.diff(cross_df["LastPrice"])
		if (max_gap < sub_df.max()):
			max_gap = sub_df.max()
			max_gap_df = cross_df.iloc[np.argmax(sub_df)]

		cross_up_df = sub_df[np.where(sub_df > 0)]
		cross_down_df = sub_df[np.where(sub_df < 0)]

		total_df += len(half_day_df)
		total_up += len(cross_up_df)
		total_down += len(cross_down_df)
		total_cross += len(sub_df)
		total_vol += day_df["Volume"].max()



		half_day_df = day_df.loc[day_df["UpdateTime"] > "13:00:00"]
		# Filter out those first / last days

		half_day_df = half_day_df.iloc[50:-50]
		if len(half_day_df) == 0:
			continue
		cross_df = half_day_df.loc[half_day_df["LastVolume"] > 0]

		sub_df = np.diff(cross_df["LastPrice"])
		if (max_gap < sub_df.max()):
			max_gap = sub_df.max()
			max_gap_df = cross_df.iloc[np.argmax(sub_df)]

		cross_up_df = sub_df[np.where(sub_df > 0)]
		cross_down_df = sub_df[np.where(sub_df < 0)]

		total_df += len(half_day_df)
		total_up += len(cross_up_df)
		total_down += len(cross_down_df)
		total_cross += len(sub_df)
		total_vol += day_df["Volume"].max()


	print "total_frame: ", total_df

	cross_rate = total_cross*1. / total_df
	print "trading rate: ", cross_rate

	print "Max gap is: ", max_gap
	print max_gap_df

	cross_up_rate = total_up*1. / total_cross
	cross_down_rate = total_down*1. / total_cross
	print "Up rate: ", cross_up_rate

	vol_rate = total_vol*1. / total_df
	print "Vol rate: ", vol_rate

	freq.append(cross_rate)
	vol.append(vol_rate)
	up.append(cross_up_rate)
	down.append(cross_down_rate)
	fair.append(1-cross_up_rate-cross_down_rate)

if __name__ == '__main__':
	draw_evaluate()
	# for fname in selected_fnames[:1]:
	# 	fpath = '../data/' + fname + ".csv"
	# 	df = read_data(fpath)
	# 	# print "get stat of ", fn 	ame
	# 	# get_stat(df)
	# 	# draw_price(df)
	# 	data, label, price = data2vector(df, mode="LSTM")
	# 	print label.shape
	# 	print label[np.where(label == 1)].shape
	# 	dump_data(data, label, price, fname)
	# draw_trend()
	# draw_freq()
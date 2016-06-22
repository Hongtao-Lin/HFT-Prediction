#!/usr/bin/python
# -*- coding:utf8 -*-

import json, re, csv
import h5py
import numpy as np

def read_data(fname):
	columns = defaultdict(list)
	with open(fname) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			y.append(int(row["lastPrize"]))
			x.append(row.values())
		x = np.array(x, dtype=float)
	return data

# return two np arrays: x and y.
def data2vec(data):
	buf = []*10
	for d in data:


	return x, y

if __name__ == '__main__':
	data = read_data("../data/.csv")
	x, y = data2vec()
	f = h5py.File("data/data.hdf5", "w")
	f["x"] = x
	f["y"] = y
	f.close()
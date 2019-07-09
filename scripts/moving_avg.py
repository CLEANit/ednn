#!/usr/bin/env python

import numpy as np
import sys

data = np.loadtxt(sys.argv[1])
window = int(sys.argv[2])

class RunningAvg:
	def __init__(self, window):
		self.items = []
		self.window = window
		self.avg = 0.

	def add(self, item):
		self.items.append(item)
		if len(self.items) > self.window:
			self.items.pop(0)
		self.avg = np.mean(self.items)

	def average(self):
		return self.avg

epochs = data[:, 0]
train_loss = data[:, 1]
val_loss = data[:, 2]


trl = RunningAvg(window)
vrl = RunningAvg(window)

for e, t, v in zip(epochs, train_loss, val_loss):
	trl.add(t)
	vrl.add(v)
	print e, trl.average(), vrl.average()

#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import h5py
import os
import cmocean
import matplotlib
import sys

matplotlib.rcParams['backend'] = "Qt4Agg"

testfiles = os.listdir('./test')
testfile = h5py.File('test/' + testfiles[0], 'r')

minval= float(sys.argv[1])
maxval = float(sys.argv[2])

tvp = np.loadtxt('output/true_vs_pred.dat')

tvp *= (maxval - minval)
tvp += minval

mean_ae = np.mean(np.abs(tvp[:,0] - tvp[:,1]))
median_ae = np.median(np.abs(tvp[:,0] - tvp[:,1]))
fig = plt.figure()
ax = fig.gca()


plt.grid()

x = np.linspace(minval, maxval, 100)
plt.plot(x,x, '--', color='k', markersize=20)

plt.hist2d(tvp[:,0], tvp[:,1], 100, norm=LogNorm(), cmap=cmocean.cm.matter)
plt.colorbar()

# plt.title('True vs. predicted distances for $H_2$', {'family': 'serif','fontsize': 15})
plt.xlabel('True', {'family': 'serif','fontsize': 12})
plt.ylabel('Predicted', {'family': 'serif','fontsize': 12})

ticklines = ax.get_xticklines() + ax.get_yticklines()
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
ticklabels = ax.get_xticklabels() + ax.get_yticklabels()

for line in ticklines:
    line.set_linewidth(3)

for line in gridlines:
    line.set_linestyle('-.')

for label in ticklabels:
    # label.set_color('r')
    label.set_fontsize('medium')


ax.text(0.0, 1.1, 'Mean absolute error: %5.5e' % (mean_ae), family='serif',
horizontalalignment='left',
verticalalignment='top',
transform=ax.transAxes)
ax.text(0.0, 1.05, 'Median abs error: %5.5e' % (median_ae), family='serif',
horizontalalignment='left',
verticalalignment='top',
transform=ax.transAxes)
plt.show()
plt.savefig('true_vs_pred.pdf')

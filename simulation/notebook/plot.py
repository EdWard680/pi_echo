#!/usr/bin/env python3
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import math

# def get_fig():
#   return plt.figure()

# def get_ax(fig, rows=1, cols=1, index=1)
#   return fig.add_subplot(int(str(rows)+str(cols)+str(index)))

def side_by_side(title=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    if title is not None: fig.suptitle(title)
    return ax1,ax2

def not_side_by_side(title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return ax


def simple_plot(ax, ydata, xdata=None, xlabel=None, ylabel=None, title=None):
    xdata = xdata or range(len(ydata))
    line, = ax.plot(xdata, ydata)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    
def plot_spectrum(ax, spectrum_list, means, f0, fpb, xlabel=None, ylabel=None, title=None):
    xmin, xmax = f0+min(means)-7, f0+max(means)+7
    xdata = np.array(range(xmin,xmax))*fpb
    lines = []
    for i,s in enumerate(spectrum_list):
        line, = ax.plot(xdata,s[x_min:x_max], label="s{}".format(i))
        lines.append(line)
    offset = max([j for i in spectrum_list for j in i])*0.01
    f0_amplitude = max([spectrum[round(f0/fpb)] for spectrum in spectrum_list])
    ax_spec.annotate('f0', xy=(f0, f0_amplitude), xytext=(f0, f0_amplitude+offset), ha="center")
    for i,(s,x) in enumerate(zip(spectrum_list,means)):
        x += f0
        ax_spec.annotate('s{}'.format(i), xy=(x, s[x]), xytext=(x, s[x]+offset), ha='center')
    ax.legend()
    fig.set_xlabel("Frequncy (Hz)")
    fig.set_ylabel("Sound Pressure (Pa)")

def maintain_bounds(xmin,xmax,ymin,ymax,x,y,u,v):
    xmin = x-u - 0.25 if x-u - 0.25 < xmin else xmin
    xmax = x+u + 0.25 if x+u + 0.25 > xmax else xmax
    ymin = y-v - 0.25 if y-v - 0.25 < ymin else ymin
    ymax = y+v + 0.25 if y+v + 0.25 > ymax else ymax
    return xmin,xmax,ymin,ymax

def plot_layout(ax, sensors, p=None, v=None, vr=None, vd=None, sv=None, sp=None):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    offset = 0.2
    
    px,py = zip(sensors)
    ax.scatter(px,py)
    if v_rads is None:
        for i,(x,y) in enumerate(sensors):
            ax.annotate("s{}".format(i), xy=(x,y), xytext=(x, y+offset), ha='center')

    if p is not None:
        px,py = p[0], [1]
        ax.scatter(px,py)
        if v is None:
            ax.annotate("P", xy=(px,py), xytext=(px, py+offset), ha='center')

    if v is not None:
        vx,vy,vu,vv = [*p,*v]
        ax.quiver(vx,vy,vu,vv, angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)
        xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,vx,vy,vu,vv)
        pos = np.dot(np.array([offset]*2), (p-v)/la.norm(p-v))
        ax.annotate("P", xy=tuple(p), xytext=tuple(pos), ha='center')

    # if vr is not None:
    #     for s,v_rad in zip(sensors,vr):



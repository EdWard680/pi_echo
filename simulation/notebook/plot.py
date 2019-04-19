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
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title is not None: fig.suptitle(title)
    return ax,fig


def simple_plot(ax, ydata, xdata=None, xlabel=None, ylabel=None, title=None):
    xdata = xdata or range(len(ydata))
    line, = ax.plot(xdata, ydata)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)



def plot_spectrum(ax, spectrum_list, means, f0, fpb, xlabel="Frequency (Hz)", ylabel="Sound Pressure", title=None):
    left, right = max([*means,0]), min([*means,0])
    xmin, xmax = f0-left-7*fpb, f0-right+7*fpb
    min_bin, max_bin = round(int(xmin)/fpb), round(int(xmax)/fpb)
    xdata = np.array(range(min_bin, max_bin))*fpb#-f0+200
    lines = []
    for i,s in enumerate(spectrum_list):
        line, = ax.plot(xdata,np.array(s[min_bin:max_bin]), label="s{}".format(i))
        lines.append(line)
    offset = max([j for i in spectrum_list for j in i])*0.01
    f0_amplitude = max([spectrum[round(f0/fpb)] for spectrum in spectrum_list])
    ax.annotate('f0', xy=(f0, f0_amplitude), xytext=(f0, f0_amplitude+offset), ha="center")
    for i,(s,x) in enumerate(zip(spectrum_list,means)):
        x = f0-x
        index = round(int(x)/fpb)
        ax.annotate('s{}'.format(i), xy=(x, s[index]), xytext=(x, s[index]+offset), ha='center')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)    
    if title is not None: ax.set_title(title)

def maintain_bounds(xmin,xmax,ymin,ymax,x,y,u,v):
    xmin = x-u - 0.25 if x-u - 0.25 < xmin else xmin
    xmax = x+u + 0.25 if x+u + 0.25 > xmax else xmax
    ymin = y-v - 0.25 if y-v - 0.25 < ymin else ymin
    ymax = y+v + 0.25 if y+v + 0.25 > ymax else ymax
    return xmin,xmax,ymin,ymax

def plot_layout(ax, sensors, p=None, v=None, vr=None, vd=None, sv=None, sp=None, xlabel=None, ylabel=None, title=None):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    offset = 0.2
    
    px,py = zip(*sensors)
    ax.scatter(px,py)
    if vr is None:
        for i,(x,y) in enumerate(sensors):
            ax.annotate("S{}".format(i), xy=(x,y), xytext=(x, y+offset), ha='center')

    if p is not None:
        px,py = p[0], p[1]
        ax.scatter(px,py)
        if v is None:
            ax.annotate("P", xy=(px,py), xytext=(px, py+offset), ha='center')

    if v is not None and p is not None:
        vx,vy,vu,vv = [*p,*v]
        ax.quiver(vx,vy,vu,vv, angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)
        xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,vx,vy,vu,vv)

        pos = p - (v)/la.norm(v)*offset
        ax.annotate("P", xy=tuple(p), xytext=tuple(pos), ha='center', , va='center')

        green_patch = mpatches.Patch(color='green', label='Ideal', alpha=0.5)
        ax.legend(handles=[green_patch], loc='upper left')

    if vr is not None:
        for i,(s,v_rad) in enumerate(zip(sensors,vr)):
            r = (p-s)/la.norm(p-s)
            proj = np.dot(r,v)*r
            # print(proj)
            vx,vy,vu,vv = [*s,*proj]
            ax.quiver(vx,vy,vu,vv, angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)
            xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,vx,vy,vu,vv)

            pos = s - proj/la.norm(proj) * offset
            ax.annotate("S{}".format(i), xy=tuple(s), xytext=tuple(pos), ha='center', va='center')

        green_patch = mpatches.Patch(color='green', label='Ideal', alpha=0.5)
        ax.legend(handles=[green_patch], loc='upper left')

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)


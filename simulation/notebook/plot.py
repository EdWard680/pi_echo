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

def plot_layout(ax, sensors, q=None, v=None, vr=None, vd=None, sq=None, sv=None, xlabel=None, ylabel=None, title=None):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    offset = 0.2
    
    px,py = zip(*sensors)
    ax.scatter(px,py)
    if vr is None:
        for i,(x,y) in enumerate(sensors):
            ax.annotate("S{}".format(i), xy=(x,y), xytext=(x, y+offset), ha='center')
            xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,x,y,0,offset)

    if q is not None:
        px,py = q[0], q[1]
        ax.scatter(px,py)
        if v is None:
            ax.annotate("Q", xy=(px,py), xytext=(px, py+offset), ha='center')
            xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,x,y,0,offset)

    if v is not None and q is not None:
        vx,vy,vu,vv = [*q,*v]
        ax.quiver(vx,vy,vu,vv, angles='xy', scale_units='xy', scale=1, color='c', alpha=0.5)
        xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,vx,vy,vu,vv)

        pos = q - (v)/la.norm(v)*offset
        ax.annotate("Q", xy=tuple(q), xytext=tuple(pos), ha='center', va='center')
        xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,q[0],q[1],pos[0]-q[0],pos[1]-q[1])

        cyan_patch = mpatches.Patch(color='cyan', label='Ideal', alpha=0.5)
        ax.legend(handles=[cyan_patch], loc='upper left')

    if vr is not None:
        for i,(s,v_rad) in enumerate(zip(sensors,vr)):
            if v_rad > 0:
                r = (q-s)/la.norm(q-s)
                proj = np.dot(r,v)*r
                vx,vy,vu,vv = [*s,*proj]
                pos = s - proj/la.norm(proj) * offset
            else:
                r = (q-s)/la.norm(q-s)
                proj = np.dot(r,v)*r
                start = s - proj
                vx,vy,vu,vv = [*start,*proj]
                pos = s + proj/la.norm(proj) * offset
            # print(proj)
            
            ax.quiver(vx,vy,vu,vv, angles='xy', scale_units='xy', scale=1, color='c', alpha=0.5)
            xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,vx,vy,vu,vv)

            
            ax.annotate("S{}".format(i), xy=tuple(s), xytext=tuple(pos), ha='center', va='center')
            xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,s[0],s[1],pos[0]-s[0],pos[1]-s[1])

        cyan_patch = mpatches.Patch(color='cyan', label='Ideal', alpha=0.5)
        ax.legend(handles=[cyan_patch], loc='upper left')

    if vd is not None:
        for i,(s,v_dop) in enumerate(zip(sensors,vd)):
            if v_dop > 0:
                r = (q-s)/la.norm(q-s)
                proj = np.dot(r,v)*r
                vx,vy,vu,vv = [*s,*proj]
                pos = s - proj/la.norm(proj) * offset
            else:
                r = (q-s)/la.norm(q-s)
                proj = np.dot(r,v)*r
                start = s - proj
                vx,vy,vu,vv = [*start,*proj]
                pos = s + proj/la.norm(proj) * offset
            # print(proj)
            
            ax.quiver(vx,vy,vu,vv, angles='xy', scale_units='xy', scale=1, color='m', alpha=0.5)
            xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,vx,vy,vu,vv)

        magenta_patch = mpatches.Patch(color='magenta', label='Simulated', alpha=0.5)
        ax.legend(handles=[cyan_patch, magenta_patch], loc='upper left')

    if sq is not None: 
        px,py = sq[0], sq[1]
        ax.scatter(px,py)
        if sv is None:
            ax.annotate("Q^", xy=(px,py), xytext=(px, py+offset), ha='center')
            xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,x,y,0,offset)
    else:
        sq = q

    if sv is not None:
        vx,vy,vu,vv = [*sq,*sv]
        ax.quiver(vx,vy,vu,vv, angles='xy', scale_units='xy', scale=1, color='m', alpha=0.5)
        xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,vx,vy,vu,vv)

        if sq is not None:
            pos = sq - (sv)/la.norm(v)*offset
            ax.annotate("Q^", xy=tuple(q), xytext=tuple(pos), ha='center', va='center')
            xmin,xmax,ymin,ymax = maintain_bounds(xmin,xmax,ymin,ymax,q[0],q[1],pos[0]-q[0],pos[1]-q[1])

        magenta_patch = mpatches.Patch(color='magenta', label='Simulated', alpha=0.5)
        ax.legend(handles=[cyan_patch, magenta_patch], loc='upper left')


    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)


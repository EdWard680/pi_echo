#!/usr/bin/env python3
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math
import itertools
from sympy.solvers.solveset import linsolve

from algorithm import *

def plot(data, shift, f0, sample_period):
	# print(f0, shift)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	

	samples = (len(data)-1)*2
	fpb = freq_per_bin(sample_period, samples)

	peak_range = doppler(f0, max_vel)

	f0_bin = round(f0 / fpb)
	bin_range = int(peak_range / fpb)

	max_peak = find_next_peak(data)
	min_peak = f0_bin if max_peak != f0_bin else None

	left_peak = f0_bin - find_next_peak(data[f0_bin:f0_bin-bin_range:-1])
	right_peak = f0_bin + find_next_peak(data[f0_bin:f0_bin+bin_range])
	max_peak = max((left_peak, right_peak),
				key=lambda x: data[x] if 0 <= x < len(data) else 0)
	# print("peaks", max_peak, f0_bin)
	peak_start = find_next_low(data[max_peak:max_peak-bin_range:-1])
	peak_end = find_next_low(data[max_peak:max_peak+bin_range])
	peak_start = peak_end = min(peak_start, peak_end) + 5
	(left_peak, right_peak) = (max_peak, f0_bin) if max_peak < f0_bin else (f0_bin, max_peak)

	plottable = data[left_peak-peak_start:right_peak+peak_end]
	# xdata = [i*(f0/len(data)) for i in range(len(plottable))]
	# xdata = [i for i in range(len(plottable))]


	# plottable = data[left_peak-peak_start:right_peak]
	# print(len(plottable), len(data))
	# # xdata = [(i+left_peak-peak_start) for i in range(len(plottable))]
	xdata = np.array(range(left_peak-peak_start, left_peak-peak_start+len(plottable)))*f0/len(data)

	line, = ax.plot(xdata, plottable)

	max_y = max(plottable)
	xpos = data.index(max_y)*(f0/len(data))
	print("peaks", max_peak, f0_bin)
	f0_pos = f0_bin*f0/len(data)
	shift_pos = max_peak*f0/len(data)
	ax.annotate('f_0', xy=(f0_pos, data[f0_bin]), xytext=(f0_pos, max_y+5), ha="center")
	ax.annotate('shift', xy=(shift_pos, data[max_peak]), xytext=(shift_pos, max_y+5), ha="center")
	# ax.text(xpos, max_y, "local max", ha="center")

	fig.show()

def plot_2d_layout(sensors, p, v):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	data = [*sensors,p]
	vel = [*p,*v]
	px, py = zip(*data)
	vx, vy, vu, vv = zip(vel)
	y_range = max(py) - min(py)
	vel_text = p-v*0.08
	print("data",data)
	ax.scatter(px,py)
	for i,(x,y) in enumerate(data):
		m = "s{}".format(i) if i < len(sensors) else "P"
		ax.annotate(m, xy=(x,y), xytext=(x,y+y_range*0.02) if i < len(sensors) else (vel_text[0], vel_text[1]), ha="center")
	ax.quiver(vx, vy, vu, vv, angles='xy', scale_units='xy', scale=1)
	fig.show()

def simulate_single_sensor(sample_freq, n, f0, p, v):
	print("Simulation")
	print("----------")
	period = 1/sample_freq
	print("Sample Frequency: ", sample_freq)
	print("Sample Period: ", period)
	print("Sample Count: ", n)
	print("Emitted Frequency: ", f0)
	print("Emitted Frequency Bin: ", f0/freq_per_bin(period, n))
	print("Starting at: ", p)
	print("Velocity vector: ", v)
	vr = np.dot(v, p / la.norm(p))
	print("Expected radial velocity: ", vr)
	print("Frequency Resolution: ", freq_per_bin(period, n), " Hz per bin")
	print("Speed Resolution: ", inv_doppler(f0, freq_per_bin(period, n)))
	print("Data Bounds: ", resolution(period, n, f0, vr), " bins")
	print("Time per FFT: ", period * n)
	print("Distance per FFT: ", period * n * la.norm(v))
	print("------------------------------")
	print("Simulating")
	print("----------")
	signal = list(signal_gen(period, n, np.array([0, 0]), p, v, f0))
	spec = freq_spectrum(signal)
	print("Frequency Bins: ", len(spec))
	u, sigma = find_shift(period, f0, spec)
	print("Frequency Shift: ", u, " (", sigma, ")")
	dopple_v, dopple_sigma = inv_doppler(f0, u), inv_doppler(f0, sigma)
	print("Doppler Velocity: ", dopple_v, " (", dopple_sigma, ")")
	err = vr - dopple_v
	print("Error: ", err, " (", err/vr*100, "%)")
	print("----------------------")
	print("Plotting")
	plot(spec, u, f0, period)

def simulate_multiple_sensors(sample_freq, n, f0, p, v, sensors):
    plot_2d_layout(sensors, p, v)
    print("Simulation")
    print("----------")
    period = 1/sample_freq
    print("Sample Frequency: ", sample_freq)
    print("Sample Period: ", period)
    print("Sample Count: ", n)
    print("Emitted Frequency: ", f0)
    print("Emitted Frequency Bin: ", f0/freq_per_bin(period, n))
    print("Starting at: ", p)
    print("Velocity vector: ", v)
    v_rads = [np.dot(v, (p - s) / la.norm(p-s) ) for s in sensors]

    print("Expected radial velocity: ", end='')
    for i,vr in enumerate(v_rads):
        print("s{}: {}".format(i, vr), end=', ' if i < len(v_rads)-1 else '\n')

    print("Frequency Resolution: ", freq_per_bin(period, n), " Hz per bin")
    print("Speed Resolution: ", inv_doppler(f0, freq_per_bin(period, n)))
    print("Data Bounds: ", *list(resolution(period, n, f0, vr) for vr in v_rads), " bins")
    print("Time per FFT: ", period * n)
    print("Distance per FFT: ", period * n * la.norm(v))
    print("------------------------------")
    print("Simulating")
    print("----------")
    print(*sensors)
    signal = [list(signal_gen(period, n, s, p, v, f0)) for s in sensors]
    spec = [freq_spectrum(sig) for sig in signal]
    print("Frequency Bins: ", len(spec[0]))
    shift_sigma = [find_shift(period, f0, sp) for sp in spec]

    print("Frequency Shift: ", end='')
    for i,sh in enumerate(shift_sigma):
        print("s{}: {} ({})".format(i, *sh), end=', ' if i < len(shift_sigma)-1 else '\n')

    dopplev_sigma = [(inv_doppler(f0, u), inv_doppler(f0, sigma)) for u,sigma in shift_sigma]

    print("Doppler Velocity: ", end='')
    for i,d in enumerate(dopplev_sigma):
        print("s{}: {} ({})".format(i, *d), end=', ' if i < len(dopplev_sigma)-1 else '\n')

    err = [v_rads[i] - dopple_v for i,(dopple_v,_) in enumerate(dopplev_sigma)]

    print("Error: ", end='')
    for i,e in enumerate(err):
        print("s{}: {} ({} %)".format(i, e, e/v_rads[i]*100), end=', ' if i < len(err)-1 else '\n')

    print("----------------------")
    print("Plotting")
    for i,s in enumerate(spec):
        plot(s, shift_sigma[i][0], f0, period)

    print("----------------------")
    print("Veloceration")
    print("----------------------")

    dopple_vs = [v[0] for v in dopplev_sigma]
    A, b, v, r = veloceration_eqs(sensors,dopple_vs)

    # print((*v,*r))
    # print(solve(A))
    # print(la.solve(A,b))
    # A = np.array([])
    # # b = np.array([r**2 - radii[-1]**2 - s[0]**2 - s[1]**2 + sensors[-1][0]**2 + sensors[-1][1]**2 for r,s in radii,sensors])
    # print(A,b)
    # x = la.solve(A,b)
    # print(x)

#!/usr/bin/env python3
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import math
import itertools
from sympy import *
from sympy.solvers.solveset import linsolve
import time

from algorithm import *
import optimizer

SAMPLE_FREQUENCY = 44100
N = 8192
F0 = 21000
P = np.array([0,0])
VELOCITY = np.array([1,1])
SENSORS = [np.array([3,5]), np.array([5,1]), np.array([1,0.5])]
METHOD = "localize"
OPTION = 0

fig_spec = plt.figure("spectrum")
ax_spec = fig_spec.add_subplot(111)

fig_2d = plt.figure("2d")
ax_2d = fig_2d.add_subplot(111)

def plot(data, shift, f0, sample_period, s, fpb):
	# print(f0, shift)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	

	samples = (len(data)-1)*2
	fpb = freq_per_bin(sample_period, samples)

	peak_range = doppler(f0, max_vel)

	# find index of shift peak
	f0_bin = round(f0 / fpb)
	bin_range = int(peak_range / fpb)

	left_peak = f0_bin - find_next_peak(data[f0_bin:f0_bin-bin_range:-1])
	right_peak = f0_bin + find_next_peak(data[f0_bin:f0_bin+bin_range])
	max_peak = max((left_peak, right_peak),
				key=lambda x: data[x] if 0 <= x < len(data) else 0)
	# get range to include in graph
	peak_start = find_next_low(data[max_peak:max_peak-bin_range:-1])
	peak_end = find_next_low(data[max_peak:max_peak+bin_range])
	peak_start = peak_end = min(peak_start, peak_end) + 5
	# assign peaks to left and right
	(left_peak, right_peak) = (max_peak, f0_bin) if max_peak < f0_bin else (f0_bin, max_peak)

	plottable = data[left_peak-peak_start:right_peak+peak_end]

	xdata = np.array(range(left_peak-peak_start, left_peak-peak_start+len(plottable)))*f0/len(data)

	line, = ax.plot(xdata, plottable)

	max_y = max(plottable)
	xpos = data.index(max_y)*fpb
	# print("peaks", max_peak, f0_bin)
	f0_pos = f0_bin*fpb
	shift_pos = max_peak*fpb
	ax.annotate('f_0', xy=(f0_pos, data[f0_bin]), xytext=(f0_pos, max_y+5), ha="center")
	ax.annotate('shift', xy=(shift_pos, data[max_peak]), xytext=(shift_pos, max_y+5), ha="center")
	# ax.text(xpos, max_y, "local max", ha="center")
	fig.suptitle("Sensor at ({},{})".format(*s))
	fig.show()

def plot_all_spectrums(spec, f0, peak_indices, x_min, x_max, fpb):
	# fig_spec = plt.figure("spectrum")
	# ax_spec = fig_spec.add_subplot(111)
	global fig_spec, ax_spec
	fig_spec.clf("spectrum")
	ax_spec = fig_spec.add_subplot(111)

	xdata = np.array(range(x_min, x_max))*fpb
	lines = []
	offset = max(spec[0])*0.01
	for i,s in enumerate(spec):
		line, = ax_spec.plot(xdata,s[x_min:x_max], label="s{}".format(i))
		lines.append(line)
	ax_spec.annotate('f0', xy=(f0, max(spec[0])), xytext=(f0, max(spec[0])+offset), ha="center")
	for i,x in enumerate(peak_indices):
		ax_spec.annotate('s{}'.format(i), xy=(x*fpb, spec[i][x]), xytext=(x*fpb, spec[i][x]+offset), ha='center')
	ax_spec.legend()
	fig_spec.suptitle("Frequency Shift due to the Doppler Effect")
	plt.xlabel("Frequncy (Hz)")
	plt.ylabel("Intensity")
	fig_spec.show()

def find_plot_chars(spectrum, f0, sample_period):
	xmin = 999999
	xmax = 0
	peaks = []
	for s in spectrum:
		samples = (len(s)-1)*2
		fpb = freq_per_bin(sample_period, samples)

		peak_range = doppler(f0, max_vel)

		# find index of shift peak
		f0_bin = round(f0 / fpb)
		bin_range = int(peak_range / fpb)

		left_peak = f0_bin - find_next_peak(s[f0_bin:f0_bin-bin_range:-1])
		right_peak = f0_bin + find_next_peak(s[f0_bin:f0_bin+bin_range])
		max_peak = max((left_peak, right_peak),
					key=lambda x: s[x] if 0 <= x < len(s) else 0)
		# get range to include in graph
		peak_start = find_next_low(s[max_peak:max_peak-bin_range:-1])
		peak_end = find_next_low(s[max_peak:max_peak+bin_range])
		peak_start = peak_end = min(peak_start, peak_end) + 5
		(left_peak, right_peak) = (max_peak, f0_bin) if max_peak < f0_bin else (f0_bin, max_peak)

		if left_peak - peak_start < xmin: xmin = left_peak - peak_start
		if right_peak + peak_start > xmax: xmax = right_peak + peak_start
		peaks.append(max_peak)
	# peaks.append(f0_bin)

	return xmin,xmax,peaks,fpb

def on_press(event):
    global SAMPLE_FREQUENCY, N, F0, P, VELOCITY, SENSORS, METHOD, OPTION
    print("====================================")
    print("====================================")
    print('Rerunning simulation using P=({},{})'.format(event.xdata, event.ydata))
    print("====================================")
    print("====================================")
    P = np.array([event.xdata,event.ydata])
    # plt.close("2d")
    # plt.close("spectrum")
    simulate_multiple_sensors(SAMPLE_FREQUENCY, N, F0, P, VELOCITY, SENSORS, METHOD, OPTION)

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def plot_2d_layout(sensors, p, v, v_rads, dopple_vs, sol, option):
	# fig_2d = plt.figure("2d")
	# ax_2d = fig_2d.add_subplot(111)
	global fig_2d, ax_2d
	fig_2d.clf("2d")
	ax_2d = fig_2d.add_subplot(111)

	xmin, xmax = ax_2d.get_xlim()
	ymin, ymax = ax_2d.get_ylim()
	xmin = -.25
	ymin = -.25

	# print(ax_2d.get_xlim())
	# print(ax_2d.get_ylim())
	data = [*sensors,p]
	px, py = zip(*data)
	y_range = max(py) - min(py)
	vel_text = p-v*0.2

	vel_vx, vel_vy, vel_vu, vel_vv = [*p,*v]
	# ax_2d.quiver(vx, vy, vu, vv, angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)
	xmin = vel_vx-vel_vu - 0.25 if vel_vx-vel_vu - 0.25 < xmin else xmin
	xmax = vel_vx+vel_vu + 0.25 if vel_vx+vel_vu + 0.25 > xmax else xmax
	ymin = vel_vy-vel_vv - 0.25 if vel_vy-vel_vv - 0.25 < ymin else ymin
	ymax = vel_vy+vel_vv + 0.25 if vel_vy+vel_vv + 0.25 > ymax else ymax

	if sol is not None:
		sol_vx, sol_vy, sol_vu, sol_vv = [*p,*sol]
		# ax_2d.quiver(vx, vy, vu, vv, angles='xy', scale_units='xy', scale=1, color='r', alpha=0.5)
		xmin = sol_vx-sol_vu - 0.25 if sol_vx-sol_vu - 0.25 < xmin else xmin
		xmax = sol_vx+sol_vu + 0.25 if sol_vx+sol_vu + 0.25 > xmax else xmax
		ymin = sol_vy-sol_vv - 0.25 if sol_vy-sol_vv - 0.25 < ymin else ymin
		ymax = sol_vy+sol_vv + 0.25 if sol_vy+sol_vv + 0.25 > ymax else ymax

	
	text_pos = []
	rad_quiver = []
	for s,v in zip(sensors,v_rads):
		proj = np.dot(v,(p-s)/la.norm(p-s))
		vx,vy,vu,vv = [*s,*proj]
		rad_quiver.append([vx,vy,vu,vv])
		# ax_2d.quiver(vx, vy, vu, vv, angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)

		xmin = vx-vu - 0.25 if vx-vu - 0.25 < xmin else xmin
		xmax = vx+vu + 0.25 if vx+vu + 0.25 > xmax else xmax
		ymin = vy-vv - 0.25 if vy-vv - 0.25 < ymin else ymin
		ymax = vy+vv + 0.25 if vy+vv + 0.25 > ymax else ymax

		text_pos.append(s-proj*0.2)

	dop_quiver = []
	for s,v in zip(sensors,dopple_vs):
		proj = np.dot(v,(p-s)/la.norm(p-s))
		vx,vy,vu,vv = [*s,*proj]
		dop_quiver.append([vx,vy,vu,vv])
		# ax_2d.quiver(vx, vy, vu, vv, angles='xy', scale_units='xy', scale=1, color='r', alpha=0.5)

		xmin = vx-vu - 0.25 if vx-vu - 0.25 < xmin else xmin
		xmax = vx+vu + 0.25 if vx+vu + 0.25 > xmax else xmax
		ymin = vy-vv - 0.25 if vy-vv - 0.25 < ymin else ymin
		ymax = vy+vv + 0.25 if vy+vv + 0.25 > ymax else ymax


	red_patch = mpatches.Patch(color='red', label='Simulated')
	green_patch = mpatches.Patch(color='green', label='Ideal')
	plt.legend(handles=[red_patch, green_patch])

	cid = fig_2d.canvas.mpl_connect('button_press_event', on_press)

	if ymin > -0.25: ymin = -0.25
	if xmin > -0.25: xmin = -0.25
	plt.xlim(xmin,xmax)
	plt.ylim(ymin,ymax)
	fig_2d.suptitle("Simulated vs. Ideal Radial Velocities")
	move_figure(fig_2d, 1000, 100)

	ax_2d.scatter(px,py)
	for i,(x,y) in enumerate(data):
		m = "s{}".format(i) if i < len(sensors) else "P"
		ax_2d.annotate(m, xy=(x,y), xytext=(tuple(text_pos[i])) if i < len(sensors) else tuple(vel_text), ha="center")
	fig_2d.show()

	if option: plt.pause(1)

	ax_2d.quiver(vel_vx, vel_vy, vel_vu, vel_vv, angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)
	fig_2d.show()

	if option: plt.pause(1)

	vx, vy, vu, vv = zip(*rad_quiver)
	ax_2d.quiver(vx, vy, vu, vv, angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)
	fig_2d.show()

	if option: plt.pause(1)

	vx, vy, vu, vv = zip(*dop_quiver)
	ax_2d.quiver(vx, vy, vu, vv, angles='xy', scale_units='xy', scale=1, color='r', alpha=0.5)
	fig_2d.show()

	if option: plt.pause(1)

	ax_2d.quiver(sol_vx, sol_vy, sol_vu, sol_vv, angles='xy', scale_units='xy', scale=1, color='r', alpha=0.5)
	fig_2d.show()


def simulate_single_sensor(sample_freq, n, f0, p, v):
	velocity_vector = v
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
	u, sigma, attenuation = find_shift(period, f0, spec)
	print("Frequency Shift: ", u, " (", sigma, ")")
	dopple_v, dopple_sigma = inv_doppler(f0, u), inv_doppler(f0, sigma)
	print("Doppler Velocity: ", dopple_v, " (", dopple_sigma, ")")
	err = vr - dopple_v
	print("Error: ", err, " (", err/vr*100, "%)")
	print("----------------------")
	print("Plotting")
	plot(spec, u, f0, period)

def simulate_multiple_sensors(sample_freq, n, f0, p, v, sensors, method="localize", option=0):
    global SAMPLE_FREQUENCY, N, F0, P, VELOCITY, SENSORS, METHOD, OPTION
    SAMPLE_FREQUENCY = sample_freq
    N = n
    F0 = f0
    P = p
    VELOCITY = v
    SENSORS = sensors
    METHOD = method
    OPTION = option
    velocity_vector = v
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

    print("Expexted distance from P: ", end='')
    for i,s in enumerate(sensors):
    	print("s{}: {}".format(i, la.norm(p-s)), end=', ' if i < len(sensors)-1 else '\n')

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
    # print(*sensors)
    signal = [list(signal_gen(period, n, s, p, v, f0)) for s in sensors]
    spec = [freq_spectrum(sig) for sig in signal]
    print("Frequency Bins: ", len(spec[0]))
    shift_sigma_att = [find_shift(period, f0, sp) for sp in spec]

    print("Frequency Shift: ", end='')
    for i,sh in enumerate(shift_sigma_att):
        print("s{}: {} ({})".format(i, *sh), end=', ' if i < len(shift_sigma_att)-1 else '\n')

    dopplev_sigma = [(inv_doppler(f0, u), inv_doppler(f0, sigma)) for u,sigma,_ in shift_sigma_att]

    attenuation = [i[2] for i in shift_sigma_att]
    print("Distance from P: ", end='')
    for i,a in enumerate(attenuation):
    	print("s{}: {}".format(i, a and 1/a), end=', ' if i < len(attenuation)-1 else '\n')

    print("Doppler Velocity: ", end='')
    for i,d in enumerate(dopplev_sigma):
        print("s{}: {} ({})".format(i, *d), end=', ' if i < len(dopplev_sigma)-1 else '\n')

    err = [v_rads[i] - dopple_v for i,(dopple_v,_) in enumerate(dopplev_sigma)]

    print("Error: ", end='')
    for i,e in enumerate(err):
        print("s{}: {} ({} %)".format(i, e, e/v_rads[i]*100), end=', ' if i < len(err)-1 else '\n')

    print("----------------------")
    print("Plotting")
    x_min, x_max, peak_indices, fpb = find_plot_chars(spec, f0, period)
    # for i,s in enumerate(spec):
    #     plot(s, shift_sigma_att[i][0], f0, period, sensors[i], fbp)
    plot_all_spectrums(spec, f0, peak_indices, x_min, x_max, fpb)
    print("----------------------")
    print("Veloceration")
    print("----------------------")

    dopple_vs = [v[0] for v in dopplev_sigma]
    
    if method == "localize":
        A, b, v, r = veloceration_eqs(sensors,dopple_vs)

        # print({r:1/a for r,a in zip(r, attenuation)})
        A = [e.subs({r:1/a for r,a in zip(r, attenuation)}) for e in A]   

        # print(linsolve(A,v))
        # print(solve(A))

        sol = linsolve(A,v)
        if sol:
            sol = np.array(sol.args[0], dtype=float)
            print("Velocity Vector: [{},{}]".format(*sol))
        else:
            print("No solution found")
            sol = None
        # print(sol)
    elif method == "gradient_descent":
        s = optimizer.Solver(sensors)
        sol = s.find_min(dopple_vs)[1]
    plot_2d_layout(sensors, p, velocity_vector, v_rads, dopple_vs, sol, option)

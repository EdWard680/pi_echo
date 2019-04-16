#!/usr/bin/env python3
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math

import algorithm

def plot(data):
	plt.plot(data)
	plt.show()

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
	plot(spec)

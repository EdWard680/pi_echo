#!/usr/bin/python3
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math

# Speed of sound
c = 343
max_vel = 30

# Returns the time over which one transform would be calculated
def fft_period(sample_period, samples):
	return samples * sample_period

# Returns the number of hertz per frequency bin in the fft
def freq_per_bin(sample_period, samples):
	return 1 / fft_period(sample_period, samples)

# Returns the frequency shifted by the doppler effect
def doppler(f0, v):
	return c / (c + v) * f0

# Returns the velocity based on the known frequency shift
def inv_doppler(f0, df):
	return df/f0*c

''' For a given set of parameters, returns the resolution with which
	the system will be able to detect the doppler effect
'''
def resolution(sample_period, samples, f0, vmax):
	return abs(int((doppler(f0, vmax) - f0) / freq_per_bin(sample_period, samples)))

''' Generates the signal received by a sensor at 'sensor' and a target object
	at 'start' moving at a 'velocity' due to the doppler effect of the
	frequency 'f0'
'''
def signal_gen(sample_period, samples, sensor, start, velocity, f0):
	t = 0
	for i in range(samples):
		r = sensor - start + velocity*t
		w = np.dot(velocity, r / la.norm(r))
		f = doppler(f0, w)
		yield math.sin(2*math.pi*f*t) + math.sin(2*math.pi*f0*t)
		t += sample_period

# Finds a local min
def find_next_low(data):
	minv = None
	for i, v in enumerate(data):
		if minv is None:
			minv = v
		elif v < minv:
			minv = v
		else:
			return i-1
	
	return len(data)

# Finds the global max starting at the first local min
def find_next_peak(data):
	maxv = 0
	ret = 0
	start = find_next_low(data)
	data = data[start:]
	for i, v in enumerate(data):
		if v > maxv:
			maxv = v
			ret = i
	
	return start + ret

''' Returns mean and standard deviation of input data
'''
def characterize_peak(data):
	sumweight = 0
	sumdata = 0
	for i, v in enumerate(data):
		sumweight += v
		sumdata += i*v
	
	mean = sumdata / sumweight
	
	variance = 0
	for i, v in enumerate(data):
		variance += v / sumweight * (mean - i)**2
	
	std_dev = math.sqrt(variance)
	
	return (mean, std_dev)

''' Finds the doppler shifted frequency in a spectrum about f0.
	Returns the mean and standard deviation
'''
def find_shift(sample_period, f0, spectrum):
	samples = (len(spectrum)-1)*2
	fpb = freq_per_bin(sample_period, samples)
	
	peak_range = doppler(f0, max_vel)
	
	f0_bin = round(f0 / fpb)
	bin_range = int(peak_range / fpb)
	
	
	# Find highest bin in search range to left and right of f0 peak
	left_peak = f0_bin - find_next_peak(spectrum[f0_bin:f0_bin-bin_range:-1])
	right_peak = f0_bin + find_next_peak(spectrum[f0_bin:f0_bin+bin_range])
	
	max_peak = max((left_peak, right_peak),
				key=lambda x: spectrum[x] if 0 <= x < len(spectrum) else 0)
	
	peak_start = find_next_low(spectrum[max_peak:max_peak-bin_range:-1])
	peak_end = find_next_low(spectrum[max_peak:max_peak+bin_range])
	
	# symmetry
	peak_start = peak_end = min(peak_start, peak_end)
	
	mean, std_dev = characterize_peak(spectrum[max_peak-peak_start:max_peak+peak_end])
	mean += max_peak-peak_start
	
	freq_mean = mean*fpb
	std_dev = std_dev*fpb
	
	return freq_mean - f0, std_dev

def freq_spectrum(signal):
	return [np.absolute(sample) for sample in np.fft.rfft(signal)]

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

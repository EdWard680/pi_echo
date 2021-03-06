#!/usr/bin/env python3
import math
import numpy as np
from numpy import linalg as la
from sympy import symbols, solve
import itertools

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
		r = start - sensor + velocity*t
		scale = 1/la.norm(r)*1.1
		w = np.dot(velocity, r / la.norm(r))
		f = doppler(f0, w)
		yield math.sin(2*math.pi*f*t)*scale + math.sin(2*math.pi*f0*t)
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
	if len(data) == 0:
		return 0, None
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
	
	if peak_start == 0:
		return 0, 2*freq_per_bin(sample_period, samples), 0
	
	mean, std_dev = characterize_peak(spectrum[max_peak-peak_start:max_peak+peak_end])
	mean += max_peak-peak_start
	
	freq_mean = mean*fpb
	std_dev = std_dev*fpb
	
	return f0 - freq_mean, std_dev, spectrum[max_peak]/spectrum[f0_bin]

def freq_spectrum(signal):
	return [np.absolute(sample) for sample in np.fft.rfft(signal)]

def veloceration_eqs(sensors, dopple_vs):
	dim = len(sensors[0])
	v = [*symbols(
			' '.join(["v{}".format(i) for i in range(dim)])
		)]
	r = [*symbols(
			' '.join(["r{}".format(i) for i,_ in enumerate(sensors)])
		)]

	s_combs = list(itertools.combinations(sensors, 2))
	v_combs = list(itertools.combinations(dopple_vs, 2))
	index_combs = list(list(itertools.combinations(range(len(sensors)), 2)))
	combs = list(zip(s_combs, v_combs, index_combs))
	# print(s_combs)
	x,y = symbols('x y')

	A = [np.dot(pi-pj,v) + vi*r[i] - vj*r[j] for ((pi,pj),(vi,vj),(i,j)) in combs]
	# A = [np.dot(pi-pj,v) + ((pi[0]-x)**2+(pi[1]-y)**2)**(1/2)*r[i] - ((pj[0]-x)**2+(pj[1]-y)**2)**(1/2)*r[j] for ((pi,pj),(vi,vj),(i,j)) in combs]
	# for a in A: print(a)
	b = np.array([0]*len(A))

	return A,b,v,r
# ri -> sqrt((x-xi)^2 + (y-yi)^2)

#!/usr/bin/env python3
import pyaudio
import sys
import time
import numpy as np
import algorithm as utils

n = 8192
f0 = 15000
rate = 44100

# instantiate PyAudio (1)
print("Instantiating PyAudio")
p = pyaudio.PyAudio()

time.sleep(10)

print("Number of devices: ", p.get_device_count())
for i in range(p.get_device_count()):
    if p.get_device_info_by_index(i)['name'] == "USB PnP Sound Device: Audio (hw:1,0)":
        print(i)
        ind = i

def audio_callback(data, frame_count=None, time_info=None, status_flags=None):
    spec = utils.freq_spectrum([float(s) for s in data])
    f0_bin = int(f0 // utils.freq_per_bin(1/rate, n))
    region = spec[f0_bin-5:f0_bin+5]
    print("f0 area(bin={}): {}".format(f0_bin, sum(region)/len(region)))
    return (data, pyaudio.paContinue)

print("Opening Stream")
# open stream (2)
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                frames_per_buffer=n,
                input_device_index=ind,
                stream_callback=audio_callback)

print("Starting to process noise")

while True:
    print("Reading data")
    time.sleep(10)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()

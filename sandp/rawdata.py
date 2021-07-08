## 1) read the binary raw_data.
## 2) smooth the raw data ..... 

# !/usr/bin/env python
from ROOT import *
from array import array
import struct
import sys
import math
import ctypes
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sandp import full_path

from .peakfinder import find_potential_peaks

from configparser import ConfigParser

cfg = ConfigParser()
# cfg.read('/home/nilab/Processor/SanDP/sandp/config/sandix.ini')
cfg.read(full_path('config/sandix.ini'))

# nsamps = int (cfg['peaks']['nsamps'])
nchs = int(cfg['peaks']['nchs'])


######################################################################

## 1)
## Read Raw Data:
## Be noted: 4 is the length of the str. it is always 4 bytes. 
def get_raw(event_number, filename, nsamps):
    length_unit = 4

    ## Seek the offset:
    def calculate_seek_number(event_number):
        if event_number == 0:
            return length_unit  ## This offset is for the Unix before the header of the first event.
        return length_unit + event_number * length_unit * 6 + event_number * nsamps * nchs * length_unit / 2
        ##        UnixTime + event_number*Header        + event_number*samples
        ##        (note: samples need to be /2 according to the CAEN manual)

    const_time = 2 ** 31  ## 31 bit counter (TRIGGER TIME TAG)
    ss = open(filename)  ## name of the raw_data

    ## Last event:
    seek_number = calculate_seek_number(event_number - 1)  ## -1 means last event
    ss.seek(seek_number + length_unit * 5)  ## 5*length_unit offset to TRIGGER TIME TAG
    pre_counter = struct.unpack('i', ss.read(4))[0] & 0xffffffff

    ## This event:
    seek_number = calculate_seek_number(event_number)
    ss.seek(seek_number + length_unit * 5)  ## 5*length_unit offset to TRIGGER TIME TAG
    counter = struct.unpack('i', ss.read(4))[0] & 0xffffffff

    ## Calculating the Trigger Time difference between LAST and THIS event:
    if counter >= pre_counter:
        delta_counter = counter - pre_counter
    else:
        delta_counter = counter - pre_counter + const_time
    MicroSec = int(delta_counter * 8e-3)  ## Convert time to Micro_seconds

    ## Filling the data: ---->
    ## ---------------------->
    channel = []  ## 2D_list: waveform of each channels.
    data = [0] * nsamps  ## 1D_list: summed waveform from all channels.

    for j in range(nchs):
        data_tmp = []
        for i in range(nsamps / 2):
            tmp = struct.unpack("i", ss.read(4))[0]
            data_tmp.append((tmp >> 16) * 2.0 / 4096)
            data_tmp.append((tmp & 0x0000ffff) * 2.0 / 4096)

        channel.append([i for i in data_tmp])
        for i in range(nsamps):
            data[i] += data_tmp[i]

    ## ---------------------->
    ## Testing:
    # print 'Length of the summed-chs WF:    ',len(data)
    # print 'Length of the individual-ch WF: ',len(channel[0])
    # print 'Number of channels in total:    ',len(channel)
    ## ---------------------->

    return data, channel, MicroSec


## 2)
## summed WF smoothing:
def smooth(origindata, meanNum=100, cover_num=5):
    # clib=ctypes.cdll.LoadLibrary("/home/nilab/Processor/SanDP/sandp/smooth/smooth.so")
    clib = ctypes.cdll.LoadLibrary(full_path("smooth/smooth.so"))
    data_smooth = (ctypes.c_double * len(origindata))()
    for i in range(len(origindata)):
        data_smooth[i] = ctypes.c_double(origindata[i])
    clib.smooth(ctypes.byref(data_smooth), ctypes.c_int(meanNum), ctypes.c_int(len(data_smooth)),
                ctypes.c_int(cover_num))

    for i in range(cover_num):
        data_smooth[i] = 0
        data_smooth[i - 1] = 0

    return data_smooth

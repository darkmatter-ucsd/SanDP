#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import struct
import os
import time
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ROOT import *
from datetime import datetime
from array import array
from rawdata import get_raw
from rawdata import smooth
from peakfinder import find_potential_peaks
from peakrefine import split_S2
from peakrefine import peak_width
from peakrefine import accurate_peaks
from peakrefine import accurate_S1
from peakrefine import accurate_S2

from peakproperty import Entropy
from peakproperty import Uniformity
from peakproperty import integral
from peakproperty import sort_area

from configparser import ConfigParser
from sandp import full_path

cfg = ConfigParser()
cfg.read(full_path('config/sandix.ini'))

nsamps = int(cfg['peaks']['nsamps'])
nchs = int(cfg['peaks']['nchs'])

s1width_lower_limit = int(cfg['peaks']['s1width_lower_limit'])
s1width_upper_limit = int(cfg['peaks']['s1width_upper_limit'])
s2width_lower_limit = int(cfg['peaks']['s2width_lower_limit'])
s2width_upper_limit = int(cfg['peaks']['s2width_upper_limit'])

nsamp_base = int(cfg['peaks']['nsamp_base'])
s1_thre_base = int(cfg['peaks']['s1_thre_base'])
s2_thre_base = int(cfg['peaks']['s2_thre_base'])
trigger_position = int(cfg['peaks']['trigger_position'])

PMTgain = [float(cfg['gains']['ch0_gain']),
           float(cfg['gains']['ch1_gain']),
           float(cfg['gains']['ch2_gain']),
           float(cfg['gains']['ch3_gain'])]

# Tree to store the data
T1 = TTree("T1", "")

# Variables for the branches
EventID = array('i', [0])
UnixTime = array('l', [0])
MicroSec = array('i', [0])
BaseLineSumSigma = array('f', [0.0])
nchannels = array("i", [4])  #Summed Baseline STD
BaseLineChannel = array("f", nchannels[0] * [0.0])  # Baseline mean per channel
BaseLineChannelSigma = array("f", nchannels[0] * [0.0])  # Baseline STD per channel
NbS1Peaks = array('i', [0])  # total nb of S1 peaks
NbS2Peaks = array('i', [0])  # total nb of S2 peaks
maxpeaks = 100  # maximum number of potential peaks
S1sPeak, S2sPeak = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak position (Time)
S2sPosX, S2sPosY = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak position (X, Y)
S1sEntropy, S2sEntropy = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak uniformity in time
S1sUniformity, S2sUniformity = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak uniformity in position
S1sWidth, S2sWidth = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak width @ 50% height
S1sLowWidth, S2sLowWidth = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak width @ 10% height
S1sRiseTime, S2sRiseTime = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak rise time
S1sDropTime, S2sDropTime = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak drop time
S1sTot, S2sTot = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])  # peak area
S1sCoin, S2sCoin = array("i", maxpeaks * [0]), array("i", maxpeaks * [0])  # number of coincidence channels
S2sPMT = array('f', maxpeaks * [0.0])  # main s2 size in each PMT

# Branches
T1.Branch("EventID", EventID, "EventID/I")
T1.Branch("UnixTime", UnixTime, "UnixTime/L")
T1.Branch("MicroSec", MicroSec, "MicroSec/I")
T1.Branch("nchannels", nchannels, "nchannels/I")
T1.Branch("BaseLineSumSigma", BaseLineSumSigma, "BaseLineSumSigma/F")
T1.Branch("BaseLineChannel", BaseLineChannel, "BaseLineChannel[nchannels]/F")
T1.Branch("BaseLineChannelSigma", BaseLineChannelSigma, "BaseLineChannelSigma[nchannels]/F")
T1.Branch("NbS1Peaks", NbS1Peaks, "NbS1Peaks/I")
T1.Branch("NbS2Peaks", NbS2Peaks, "NbS2Peaks/I")
T1.Branch("S2sPosX", S2sPosX, "S2sPosX[NbS2Peaks]/F")
T1.Branch("S2sPosY", S2sPosY, "S2sPosY[NbS2Peaks]/F")
T1.Branch("S1sPeak", S1sPeak, "S1sPeak[NbS1Peaks]/F")
T1.Branch("S2sPeak", S2sPeak, "S2sPeak[NbS2Peaks]/F")
T1.Branch("S1sWidth", S1sWidth, "S1sWidth[NbS1Peaks]/F")
T1.Branch("S1sLowWidth", S1sLowWidth, "S1sLowWidth[NbS1Peaks]/F")
T1.Branch("S2sWidth", S2sWidth, "S2sWidth[NbS2Peaks]/F")
T1.Branch("S2sLowWidth", S2sLowWidth, "S2sLowWidth[NbS2Peaks]/F")
T1.Branch("S1sRiseTime", S1sRiseTime, "S1sRiseTime[NbS1Peaks]/F")
T1.Branch("S2sRiseTime", S2sRiseTime, "S2sRiseTime[NbS2Peaks]/F")
T1.Branch("S1sDropTime", S1sDropTime, "S1sDropTime[NbS1Peaks]/F")
T1.Branch("S2sDropTime", S2sDropTime, "S2sDropTime[NbS2Peaks]/F")
T1.Branch("S1sTot", S1sTot, "S1sTot[NbS1Peaks]/F")
T1.Branch("S2sTot", S2sTot, "S2sTot[NbS2Peaks]/F")
T1.Branch("S1sCoin", S1sCoin, "S1sCoin[NbS1Peaks]/I")
T1.Branch("S2sCoin", S2sCoin, "S2sCoin[NbS2Peaks]/I")
T1.Branch("S1sEntropy", S1sEntropy, "S1sEntropy[NbS1Peaks]/F")
T1.Branch("S2sEntropy", S2sEntropy, "S2sEntropy[NbS2Peaks]/F")
T1.Branch("S1sUniformity", S1sUniformity, "S1sUniformity[NbS1Peaks]/F")
T1.Branch("S2sUniformity", S2sUniformity, "S2sUniformity[NbS2Peaks]/F")
T1.Branch('S2sPMT', S2sPMT, 'S2sPMT[nchannels]/F')


# processing the raw_data:
def process(filename, outpath):
    outfile = TFile(outpath + '/' + filename[-26:-4] + '.root', "RECREATE")

    # Total number of events:
    infile = open(filename)
    # go to the last event to check the event counter (ID)
    infile.seek(-4 * (nchs * nsamps / 2 + 2), os.SEEK_END)
    totN = (struct.unpack('i', infile.read(4))[0] & 0x00ffffff) + 1
    print
    'Total number of event in processing: ', totN

    # Event time:
    infile.seek(0)
    HeaderTime = struct.unpack('i', infile.read(4))[0]
    HeaderTime = HeaderTime - 7 * 60 * 60  ## convert UTC time to SD time.
    print
    'Data taking time: ', datetime.utcfromtimestamp(HeaderTime).strftime('%Y-%m-%d %H:%M:%S')

    # Define variables to calculate remaining time:
    UnixTime[0] = HeaderTime  # Header Reference Time
    t_passed = 0
    t_left = 0
    time_tol = 0
    time_startc = time.time()
    Time_all = HeaderTime * 1000000  ## To MicroSec

    # Looping all selected events:
    ## ===========================>
    ## ===========================>
    for event_number in range(1, totN):
        EventID[0] = event_number

        ## print '------------------------------------------------------- ',event_number

        ## accumulating running time:
        t_passed += time.time() - time_startc
        time_startc = time.time()
        if (event_number % 500 == 0) and (event_number != 0):
            t_left = t_passed / float(event_number) * (totN - event_number)
            print("Job progress : " + str(float(event_number) / float(totN) * 100)
                  + "% Time passed/left: " + str(t_passed)
                  + "/" + str(t_left) + " sec ")

        ## Go to rawdata !!!:
        data, channel, Micro = get_raw(event_number, filename)
        data_smooth = smooth(data)
        Time_all += Micro  ## In MicroSec
        UnixTime[0] = int(Time_all / 1000000)  ## Back to Sec
        MicroSec[0] = Time_all % 1000000  # MicroSec

        ## Baseline calculation:
        BaseLineSumSigma[0] = np.std(data[:nsamp_base])
        #print('BaseLineChannelSigma: %f' % BaseLineChannelSigma[0])

        ## Find potential S1 S2:
        S1_potential = find_potential_peaks(data_smooth,
                                            s1width_lower_limit,
                                            s1width_upper_limit,
                                            max(0.001, s1_thre_base * BaseLineSumSigma[0]))
        S2_potential = find_potential_peaks(data_smooth,
                                            s2width_lower_limit,
                                            s2width_upper_limit,
                                            max(0.001, s2_thre_base * BaseLineSumSigma[0]))

        #print('S1 potential number: %d\n' % len(S1_potential))
        # print('S2 potential number: %d' %len(S2_potential))

        ## accurate S1, S2:
        S2_split = split_S2(data_smooth, S2_potential, 0.1, 1. / 5)
        S1, S2 = accurate_peaks(data_smooth, S1_potential, S2_split, s1width_upper_limit)
        S1, S2_temp = accurate_S1(data_smooth, S1, S2, s1width_upper_limit, nearestS1=400, distanceS1=40)
        S2 += S2_temp
        S2 = accurate_S2(S2)

        ## print S1
        ## print S2
        # print('S1: ', S1)
        # print('S2: ', S2)

        ## Number of S1 and S2 peaks:
        NbS1Peaks[0] = len(S1)
        NbS2Peaks[0] = len(S2)

        if NbS1Peaks[0] > 100:
            NbS1Peaks[0] = 100
        if NbS2Peaks[0] > 100:
            NbS2Peaks[0] = 100

        ## Baseline for each channel:
        for i in range(len(channel)):
            BaseLineChannel[i] = np.mean(channel[i][:nsamp_base])
            BaseLineChannelSigma[i] = np.std(channel[i][:nsamp_base])

        ## Peak Area to Generate S1 and S2 sort index
        S1s = []
        S2s = []
        for i in range(len(channel)):
            S1s.append(integral(S1, channel[i], BaseLineChannel[i], PMTgain[i]))
            S2s.append(integral(S2, channel[i], BaseLineChannel[i], PMTgain[i]))

        S1sTot_tmp = [0] * len(S1)
        S2sTot_tmp = [0] * len(S2)

        for i in range(len(S2)):
            for j in range(len(channel)):
                S2sTot_tmp[i] += S2s[j][i]
        for i in range(len(S1)):
            for j in range(len(channel)):
                S1sTot_tmp[i] += S1s[j][i]

        ## S1 and S2 sort index:
        S1s_Key = sort_area(S1sTot_tmp)
        S2s_Key = sort_area(S2sTot_tmp)

        ## Defining S2sTot and S1sTot:
        for i in range(NbS1Peaks[0]):
            S1sTot[i] = S1sTot_tmp[S1s_Key[i]]

        for i in range(NbS2Peaks[0]):
            S2sTot[i] = S2sTot_tmp[S2s_Key[i]]

        ## Peak Width and time positions:
        for i in range(NbS1Peaks[0]):
            peak = peak_width(data_smooth, 0.5, S1[S1s_Key[i]])
            peakhigh = peak_width(data_smooth, 0.9, S1[S1s_Key[i]])
            S1sWidth[i] = peak[2] - peak[0]
            S1sPeak[i] = peak[1]
            peaklow = peak_width(data_smooth, 0.1, S1[S1s_Key[i]])
            S1sLowWidth[i] = peaklow[2] - peaklow[0]
            S1sRiseTime[i] = peakhigh[0] - peaklow[0]
            S1sDropTime[i] = peaklow[2] - peakhigh[2]

        for i in range(NbS2Peaks[0]):
            peak = peak_width(data_smooth, 0.5, S2[S2s_Key[i]])
            peakhigh = peak_width(data_smooth, 0.9, S2[S2s_Key[i]])
            S2sWidth[i] = peak[2] - peak[0]
            S2sPeak[i] = peak[1]
            peaklow = peak_width(data_smooth, 0.1, S2[S2s_Key[i]])
            S2sLowWidth[i] = peaklow[2] - peaklow[0]
            S2sRiseTime[i] = peakhigh[0] - peaklow[0]
            S2sDropTime[i] = peaklow[2] - peakhigh[2]

        ## Peak Entropy for noise rejection:
        for i in range(NbS1Peaks[0]):
            S1sEntropy[i] = Entropy(nchannels[0], channel, BaseLineChannel, S1[S1s_Key[i]], BaseLineChannelSigma)
        for i in range(NbS2Peaks[0]):
            S2sEntropy[i] = Entropy(nchannels[0], channel, BaseLineChannel, S2[S2s_Key[i]], BaseLineChannelSigma)

        ## Peak X Y Position Uniformity and Reconstruction:
        for i in range(NbS2Peaks[0]):
            if (S2sTot[i] > 10):
                S2sPosX[i] = (S2s[0][S2s_Key[i]] + S2s[2][S2s_Key[i]] - S2s[1][S2s_Key[i]] - S2s[3][S2s_Key[i]]) / \
                             S2sTot[i]
                S2sPosY[i] = (S2s[0][S2s_Key[i]] - S2s[2][S2s_Key[i]] + S2s[1][S2s_Key[i]] - S2s[3][S2s_Key[i]]) / \
                             S2sTot[i]

        # Main S2 size in each PMT
        for i in range(len(channel)):
            if len(S2s_Key) > 0:
                S2sPMT[i] = S2s[i][S2s_Key[0]]

        ## S1s and S2s coincidence levels:
        for i in range(NbS2Peaks[0]):
            for ich in range(nchannels[0]):
                if (S2s[ich][S2s_Key[i]] > 0.5):
                    S2sCoin[i] += 1

        for i in range(NbS1Peaks[0]):
            for ich in range(nchannels[0]):
                if (S1s[ich][S1s_Key[i]] > 0.5):
                    S1sCoin[i] += 1

        ## Peak Distribution Uniformity:
        for i in range(NbS1Peaks[0]):
            S1sUniformity[i] = Uniformity(nchannels[0], PMTgain, channel, BaseLineChannel, S1[S1s_Key[i]],
                                          threshold=0.3)

        for i in range(NbS2Peaks[0]):
            S2sUniformity[i] = Uniformity(nchannels[0], PMTgain, channel, BaseLineChannel, S2[S2s_Key[i]],
                                          threshold=0.3)

        ## Filling the Tree:
        T1.Fill()

    infile.close()
    T1.Write()
    outfile.Close()

"""
get event property
"""
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
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
from .rawdata import get_raw
from .rawdata import smooth
from .peakfinder import find_potential_peaks
from .peakrefine import split_S2
from .peakrefine import peak_width
from .peakrefine import accurate_peaks
from .peakrefine import accurate_S1
from .peakrefine import accurate_S2
from .peakproperty import Entropy
from .peakproperty import Uniformity
from .peakproperty import integral
from .peakproperty import sort_area
from configparser import ConfigParser
from sandp import full_path


class Event(object):
    """
    Event class stores all the properties of event
    """

    def __init__(self, event_number, file_name, config):
        self.EventID = event_number
        self.file_name = file_name
        self.data_raw, self.channel, self.MicroSec = get_raw(self.EventID, self.file_name)
        self.data_smooth = smooth(self.data_raw)
        self.get_config(config)

    def get_config(self, cfg):  # TODO: don't need to initiate for each event, but for each run instead
        """
        help __init__ get properties from configuration file
        """
        self.nchannels = int(cfg['peaks']['nchs'])
        # width requirement
        self.s1width_lower_limit = int(cfg['peaks']['s1width_lower_limit'])
        self.s1width_upper_limit = int(cfg['peaks']['s1width_upper_limit'])
        self.s2width_lower_limit = int(cfg['peaks']['s2width_lower_limit'])
        self.s2width_upper_limit = int(cfg['peaks']['s2width_upper_limit'])

        self.nsamps = int(cfg['peaks']['nsamps'])
        self.nsamp_base = int(cfg['peaks']['nsamp_base'])

        self.s1_thre_base = int(cfg['peaks']['s1_thre_base'])
        self.s2_thre_base = int(cfg['peaks']['s2_thre_base'])

        self.trigger_position = int(cfg['peaks']['trigger_position'])
        self.PMTgain = [float(cfg['gains']['ch0_gain']),
                        float(cfg['gains']['ch1_gain']),
                        float(cfg['gains']['ch2_gain']),
                        float(cfg['gains']['ch3_gain'])]

    def baseline(self):
        """
        get information for baseline

        """
        self.BaseLineSumSigma = np.std(self.data_raw[:self.nsamp_base])
        # Baseline for each channel:
        self.BaseLineChannel, self.BaseLineChannelSigma = [], []
        for i in range(len(self.channel)):
            self.BaseLineChannel.append(np.mean(self.channel[i][:self.nsamp_base]))
            self.BaseLineChannelSigma.append(np.std(self.channel[i][:self.nsamp_base]))

    def boundary(self):
        """
        get S1, S2 information, including

        - S1: list of list of left and right boundaries of S1
        - S2: list of list of left and right boundaries of S2
        - NbS1Peaks: number of S1 peaks
        - NbS2Peaks: number of S2 peaks
        """
        S1_potential = find_potential_peaks(self.data_smooth,
                                            self.s1width_lower_limit,
                                            self.s1width_upper_limit,
                                            max(0.001, self.s1_thre_base * self.BaseLineSumSigma))
        S2_potential = find_potential_peaks(self.data_smooth,
                                            self.s2width_lower_limit,
                                            self.s2width_upper_limit,
                                            max(0.001, self.s2_thre_base * self.BaseLineSumSigma))

        # print('S1 potential number: %d\n' % len(S1_potential))
        # print('S2 potential number: %d' %len(S2_potential))

        # accurate S1, S2:
        S2_split = split_S2(self.data_smooth, S2_potential, 0.1, 1./5)
        S1, S2 = accurate_peaks(self.data_smooth, S1_potential, S2_split, self.s1width_upper_limit)
        S1, S2_temp = accurate_S1(self.data_smooth, S1, S2, self.s1width_upper_limit, nearestS1=400, distanceS1=40)
        S2 += S2_temp
        S2 = accurate_S2(S2)

        # put S1 S2 related info
        self.S1 = S1
        self.S2 = S2
        self.NbS1Peaks = min(len(self.S1), 100)  # number of peaks no more than 100
        self.NbS2Peaks = min(len(self.S2), 100)  # TODO: is it necessary?

    def size(self):
        """
        get s1 s2 size

        - S1sTot: list of S1 size, indexed by size
        - S2sTot: list of S2 size, indexed by size
        - S1s_Key: Keys for S1 to be ordered by size
        - S2s_Key: Keys for S2 to be ordered by size
        """
        S1s = []
        S2s = []
        for i in range(len(self.channel)):
            S1s.append(integral(self.S1, self.channel[i], self.BaseLineChannel[i], self.PMTgain[i]))
            S2s.append(integral(self.S2, self.channel[i], self.BaseLineChannel[i], self.PMTgain[i]))

        self.S1s = S1s
        self.S2s = S2s

        S1sTot_tmp = [0] * len(self.S1)
        S2sTot_tmp = [0] * len(self.S2)

        for i in range(len(self.S2)):
            for j in range(len(self.channel)):
                S2sTot_tmp[i] += S2s[j][i]
        for i in range(len(self.S1)):
            for j in range(len(self.channel)):
                S1sTot_tmp[i] += S1s[j][i]

        # S1 and S2 sort index:
        self.S1s_Key = sort_area(S1sTot_tmp)
        self.S2s_Key = sort_area(S2sTot_tmp)

        # Defining S2sTot and S1sTot:
        self.S1sTot = [S1sTot_tmp[key] for key in self.S1s_Key]
        self.S2sTot = [S2sTot_tmp[key] for key in self.S2s_Key]

    def peak_width(self):
        """
        get peak width and rise/drop time information
        """
        ## Peak Width and time positions:
        self.S1sWidth = []
        self.S1sPeak = []
        self.S1sLowWidth = []
        self.S1sRiseTime = []
        self.S1sDropTime = []
        self.S2sWidth = []
        self.S2sPeak = []
        self.S2sLowWidth = []
        self.S2sRiseTime = []
        self.S2sDropTime = []
        for i in range(self.NbS1Peaks):
            peak = peak_width(self.data_smooth, 0.5, self.S1[self.S1s_Key[i]])
            self.S1sWidth.append(peak[2] - peak[0])
            self.S1sPeak.append(peak[1])
            peaklow = peak_width(self.data_smooth, 0.1, self.S1[self.S1s_Key[i]])
            self.S1sLowWidth.append(peaklow[2] - peaklow[0])
            self.S1sRiseTime.append(peak[0] - peaklow[0])
            self.S1sDropTime.append(peaklow[2] - peak[2])

        for i in range(self.NbS2Peaks):
            peak = peak_width(self.data_smooth, 0.5, self.S2[self.S2s_Key[i]])
            self.S2sWidth.append(peak[2] - peak[0])
            self.S2sPeak.append(peak[1])
            peaklow = peak_width(self.data_smooth, 0.1, self.S2[self.S2s_Key[i]])
            self.S2sLowWidth.append(peaklow[2] - peaklow[0])
            self.S2sRiseTime.append(peak[0] - peaklow[0])
            self.S2sDropTime.append(peaklow[2] - peak[2])

    def prepare_entropy(self):
        """
        get entropy for s1 and s2 peaks
        """
        # Peak Entropy for noise rejection:
        self.S1sEntropy, self.S2sEntropy = [], []
        for i in range(self.NbS1Peaks):
            self.S1sEntropy.append(Entropy(self.nchannels, self.channel, self.BaseLineChannel,
                                           self.S1[self.S1s_Key[i]], self.BaseLineChannelSigma))
        for i in range(self.NbS2Peaks):
            self.S2sEntropy.append(Entropy(self.nchannels, self.channel, self.BaseLineChannel,
                                           self.S2[self.S2s_Key[i]], self.BaseLineChannelSigma))

    def get_positions(self):
        """
        get xy position for event
        """
        # Peak X Y Position Uniformity and Reconstruction:
        self.S2sPosX = np.full_like(self.S2sTot, np.nan, dtype=np.double)
        self.S2sPosY = np.full_like(self.S2sTot, np.nan, dtype=np.double)
        for i in range(self.NbS2Peaks):
            if (self.S2sTot[i] > 10):
                self.S2sPosX[i] = (self.S2s[0][self.S2s_Key[i]] + self.S2s[2][self.S2s_Key[i]]
                                   - self.S2s[1][self.S2s_Key[i]] - self.S2s[3][self.S2s_Key[i]]) / self.S2sTot[i]
                self.S2sPosY[i] = (self.S2s[0][self.S2s_Key[i]] - self.S2s[2][self.S2s_Key[i]]
                                   + self.S2s[1][self.S2s_Key[i]] - self.S2s[3][self.S2s_Key[i]]) / self.S2sTot[i]

    def main_s2_in_pmt(self):
        """
        get main s2 size in each PMT
        """
        x = np.arange(self.nchannels)
        self.S2sPMT = np.full_like(x, np.nan, dtype=np.double)
        if self.NbS2Peaks > 0:
            for i in range(self.nchannels):
                # print('self.S2s: ', self.S2s)
                # print('self.S2s_Key[0]: ', self.S2s_Key[0])
                #if len(self.S2s_Key) > 0:
                #if self.S2sTot[i] > 10:  # FIX: tmp condition
                self.S2sPMT[i] = self.S2s[i][self.S2s_Key[0]]

    def coincidence(self):
        """
        S1s and S2s coincidence levels
        """
        self.S2sCoin = []
        for i in range(self.NbS2Peaks):
            coin = 0
            for idx in range(self.nchannels):
                if (self.S2s[idx][self.S2s_Key[i]] > 0.5):  # TODO: different size for S1 and S2?
                    coin += 1
            self.S2sCoin.append(coin)

        self.S1sCoin = []
        for i in range(self.NbS1Peaks):
            coin = 0
            for idx in range(self.nchannels):
                if (self.S1s[idx][self.S1s_Key[i]] > 0.5):
                    coin += 1
            self.S1sCoin.append(coin)

    def uniformity(self):
        self.S1sUniformity, self.S2sUniformity = [], []
        for i in range(self.NbS1Peaks):
            self.S1sUniformity.append(Uniformity(self.nchannels, self.PMTgain, self.channel, self.BaseLineChannel,
                                                 self.S1[self.S1s_Key[i]], threshold=0.3))
        for i in range(self.NbS2Peaks):
            self.S2sUniformity.append(Uniformity(self.nchannels, self.PMTgain, self.channel, self.BaseLineChannel,
                                                 self.S2[self.S2s_Key[i]], threshold=0.3))
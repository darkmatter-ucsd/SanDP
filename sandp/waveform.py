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
from .event import Event

cfg = ConfigParser()
# cfg.read('/home/nilab/Processor/SanDP/sandp/config/sandix.ini')
cfg.read(full_path('config/sandix.ini'))


def drawWF(evt, fname, savepath=False):
    # get event information
    event = Event(evt, fname, cfg)
    event.baseline()
    event.boundary()
    event.size()
    event.peak_width()

    rawbaseline = np.mean(event.data_raw[0:event.nsamp_base])
    dat_raw_temp = []
    for i in range(len(event.data_raw)):  # TODO: simply it
        dat_raw_temp.append(rawbaseline - event.data_raw[i])  # TODO: where does calvariable reverse raw data?

    samp_len = event.nsamps  # from sample to micro-second
    sams = np.linspace(0, samp_len, samp_len)
    miny = min(min(event.channel[0]), min(event.channel[1]), min(event.channel[2]), min(event.channel[3]))
    maxy = max(max(event.channel[0]), max(event.channel[1]), max(event.channel[2]), max(event.channel[3]))

    # individul WF:
    plt.figure(figsize=(8, 7))
    plt.subplot(411)
    # print(len(sams), len(channel[0]))
    plt.plot(sams, event.channel[0], color='black', linestyle='-', linewidth=.5, label='')
    plt.xlim(-10, samp_len + 10)
    plt.ylim(miny, maxy)
    plt.xlabel('', fontsize=0)
    plt.ylabel('ch0 [V]', fontsize=10)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.axvline(6500, color='magenta', linestyle='--', linewidth = 1)

    plt.subplot(412)
    plt.plot(sams, event.channel[1], color='black', linestyle='-', linewidth=.5, label='')
    plt.xlabel('Samples [4ns]', fontsize=10)
    plt.ylabel('ch1 [V]', fontsize=10)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(-10, samp_len + 10)
    plt.ylim(miny, maxy)
    # plt.axvline(6500, color='magenta', linestyle='--', linewidth = 1)

    plt.subplot(413)
    plt.plot(sams, event.channel[2], color='black', linestyle='-', linewidth=.5, label='')
    plt.xlim(-10, samp_len + 10)
    plt.ylim(miny, maxy)
    plt.xlabel('Samples [4ns]', fontsize=10)
    plt.ylabel('ch2 [V]', fontsize=10)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.axvline(6500, color='magenta', linestyle='--', linewidth = 1)

    plt.subplot(414)
    plt.plot(sams, event.channel[3], color='black', linestyle='-', linewidth=.5, label='')
    plt.xlim(-10, samp_len + 10)
    plt.ylim(miny, maxy)
    plt.xlabel('Samples [4ns]', fontsize=10)
    plt.ylabel('ch3 [V]', fontsize=10)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.axvline(6500, color='magenta', linestyle='--', linewidth = 1)

    # plot summed WF
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.plot(sams, dat_raw_temp,
             color='black', ls='-',
             lw=0.5, label='Summed Waveform')
    plt.plot(sams, event.data_smooth, color='springgreen',
             ls='-', lw=1., alpha=1, label='Smoothed Waveform')
    plt.xlim(0, samp_len)
    # plt.xlim(4900,  5100)
    # plt.ylim(0,  0.5)
    plt.xlabel('Samples', fontsize=15)
    plt.ylabel('Amp [V]', fontsize=15)
    plt.grid(True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # plot baseline for s1 and s2
    plt.axhline(event.s1_thre_base * event.BaseLineSumSigma,
                color='magenta', linestyle='--', linewidth=1.,
                label='S1 Threshold')
    plt.axhline(event.s2_thre_base * event.BaseLineSumSigma,
                color='yellow', linestyle='--', linewidth=1.,
                label='S2 Threshold')
    # S1:
    if len(event.S1) > 0:
        for p1 in range(len(event.S1)):
            if p1 == 0:
                ax.axvspan(event.S1[p1][0], event.S1[p1][1], alpha=0.4, color='magenta', label='S1')
            else:
                ax.axvspan(event.S1[p1][0], event.S1[p1][1], alpha=0.4, color='magenta', label='')
        # indicate the largest s1
        plt.text(event.S1sPeak[0] + 200, event.data_smooth[int(event.S1sPeak[0])],
                 'S1[0]\n\n%.1f PE' % (event.S1sTot[0]),
                 fontsize=10, color='blue',
                 fontweight='bold',
                 zorder=100)

    # S2:
    if len(event.S2) > 0:
        for p2 in range(len(event.S2)):
            if p2 == 0:
                # ax.axvspan(peaklow[p2][0], peaklow[p2][2], alpha=0.4, facecolor='magenta', edgecolor = 'black', label = 'S2')
                ax.axvspan(event.S2[p2][0], event.S2[p2][1],
                           alpha=0.4, facecolor='yellow',
                           edgecolor='black', label='S2')
                # print ('========> ',peaklow[p2][0], peaklow[p2][2])
            else:
                ax.axvspan(event.S2[p2][0], event.S2[p2][1],
                           alpha=0.4, facecolor='yellow',
                           edgecolor='black', label='')
                # print ('========> ',peaklow[p2][0], peaklow[p2][2])
        # indicate the largest s2
        plt.text(event.S2sPeak[0] + 200, event.data_smooth[int(event.S2sPeak[0])],
                 'S2[0]\n\n%.1f PE' % (event.S2sTot[0]),
                 fontsize=10, color='blue',
                 fontweight='bold',
                 zorder=100)

    plt.title(fname[-19:-4] + ' event ' + str(evt))  # TODO: change the numbers to split
    plt.legend(loc='best', shadow=True, fontsize=12.5)
    plt.ylim(0, )
    if savepath:
        name = fname[-26:-4] + '_event_' + str(evt) + '.png'
        plt.savefig(os.path.join(savepath, name), dpi=300)
        plt.close()
    else:
        plt.show(block=False)
        raw_input("Hit Enter To Close")
        plt.close()

#
# def drawWF(evt, fname, savepath=False):
#     dat_raw, channel, MicroSec = get_raw(evt, fname)
#     data_smooth = smooth(dat_raw)
#
#     # ----------->
#     rawbaseline = np.mean(dat_raw[0:nsamp_base])
#     dat_raw_temp = []
#     for i in range(len(dat_raw)):  # TODO: simply it
#         dat_raw_temp.append(rawbaseline - dat_raw[i])
#     # ----------->
#
#     ## Baseline:
#     BaseLineSumSigma = array('f', [0.0])
#     BaseLineSumSigma[0] = np.std(dat_raw_temp[:nsamp_base])
#     ##=== find potential S1 S2 =====
#     S1_potential = find_potential_peaks(data_smooth[:],
#                                         s1width_lower_limit,
#                                         s1width_upper_limit,
#                                         max(0.001, s1_thre_base * BaseLineSumSigma[0]))
#
#     S2_potential = find_potential_peaks(data_smooth,
#                                         s2width_lower_limit,
#                                         s2width_upper_limit,
#                                         max(0.001, s2_thre_base * BaseLineSumSigma[0]))
#     # print('S1 edges: ', S1_potential)
#     # print('S2 edges: ', S2_potential)
#     print('Event number: ', evt)
#     print('Potential S1 number: ', len(S1_potential))
#     # print('Potential S2 number: ', len(S2_potential))
#
#     S2_split = split_S2(data_smooth, S2_potential, 0.1, 1. / 5)
#     # print('S2_split', S2_split)
#     S1, S2 = accurate_peaks(data_smooth, S1_potential, S2_split, s1width_upper_limit)
#     # print('accurate_peaks')
#     # print('S1: ', S1)
#     # print('S2: ', S2)
#     S1, S2_temp = accurate_S1(data_smooth, S1, S2, s1width_upper_limit, nearestS1=400, distanceS1=40)
#     # print('s1width_upper_limit', s1width_upper_limit)
#     S2 += S2_temp
#     # print('accurate_S1')
#     # print('S1: ', S1)
#     # print('S2: ', S2)
#     S2 = accurate_S2(S2)
#     # print('accurate_S2')
#     # print('S1: ', S1)
#     # print('S2: ', S2)
#
#     # print('S1 edges:', S1)
#     # print('S2 edges:', S2)
#     # print("S1 number: %d" %len(S1))
#     # print("S2 number: %d" %len(S2))
#
#     ################ Find the Largest S1 & S2 ======>>>
#     # ------------------------------------------------------------------------------------>
#     # ------------------------------------------------------------------------------------>
#     ## Number of S1 and S2 peaks:
#     NbS1Peaks = array('i', [0])
#     NbS2Peaks = array('i', [0])
#     NbS1Peaks[0] = len(S1)
#     NbS2Peaks[0] = len(S2)
#
#     if NbS1Peaks[0] > 100:
#         NbS1Peaks[0] = 100
#     if NbS2Peaks[0] > 100:
#         NbS2Peaks[0] = 100
#
#     ## Baseline for each channel:
#     nchannel = array("i", [4])
#     BaseLineChannel = array("f", nchannel[0] * [0.0])
#     BaseLinechanneligma = array("f", nchannel[0] * [0.0])
#
#     for i in range(len(channel)):
#         BaseLineChannel[i] = np.mean(channel[i][:nsamp_base])
#         BaseLinechanneligma[i] = np.std(channel[i][:nsamp_base])
#
#     ## Peak Area to Generate S1 and S2 sort index：
#     S1s = []
#     S2s = []
#     for i in range(len(channel)):
#         S1s.append(integral(S1, channel[i], BaseLineChannel[i], PMTgain[i]))
#         S2s.append(integral(S2, channel[i], BaseLineChannel[i], PMTgain[i]))
#
#     S1sTot_tmp = [0] * len(S1)
#     S2sTot_tmp = [0] * len(S2)
#
#     for i in range(len(S2)):
#         for j in range(len(channel)):
#             S2sTot_tmp[i] += S2s[j][i]
#     for i in range(len(S1)):
#         for j in range(len(channel)):
#             S1sTot_tmp[i] += S1s[j][i]
#
#     # print s2 size
#     # print('S2 area: ', S2sTot_tmp)
#
#     ## S1 and S2 sort index:
#     S1s_Key = sort_area(S1sTot_tmp)
#     S2s_Key = sort_area(S2sTot_tmp)
#
#     # print s2 size
#     # print('S2 area: ', S2sTot_tmp)
#
#     ## Defining S2sTot and S1sTot:
#     maxpeaks = 100
#     S1sTot, S2sTot = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])
#     for i in range(NbS1Peaks[0]):
#         S1sTot[i] = S1sTot_tmp[S1s_Key[i]]
#
#     for i in range(NbS2Peaks[0]):
#         S2sTot[i] = S2sTot_tmp[S2s_Key[i]]
#
#     S1sPeak, S2sPeak = array("f", maxpeaks * [0.0]), array("f", maxpeaks * [0.0])
#     for i in range(NbS1Peaks[0]):
#         peak = peak_width(data_smooth, 0.5, S1[S1s_Key[i]])
#         S1sPeak[i] = peak[1]
#
#     peaklow = []
#     for i in range(NbS2Peaks[0]):
#         peak = peak_width(data_smooth, 0.5, S2[S2s_Key[i]])
#         S2sPeak[i] = peak[1]
#
#         peaklow.append(peak_width(data_smooth, 0.1, S2[S2s_Key[i]]))
#         # S2sLowWidth[i] = peaklow[2]-peaklow[0]
#
#     ######################################################################################################:
#     ##############################     Make Drawings      ################################################:
#     ######################################################################################################:
#     samp_len = nsamps  ## from sample to micro-second
#     sams = np.linspace(0, samp_len, samp_len)
#
#     miny = min(min(channel[0]), min(channel[1]), min(channel[2]), min(channel[3]))
#     maxy = max(max(channel[0]), max(channel[1]), max(channel[2]), max(channel[3]))
#     ## individul WF:
#     fig_individual = plt.figure(figsize=(8, 7))
#     plt.subplot(411)
#
#     plt.xlim(-10, samp_len + 10)
#     plt.ylim(miny, maxy)
#
#     plt.xlabel('', fontsize=0)
#     plt.ylabel('ch0 [V]', fontsize=10)
#     plt.grid(True)
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     # print(len(sams), len(channel[0]))
#     plt.plot(sams, channel[0], color='black', linestyle='-', linewidth=.5, label='')
#     # plt.axvline(6500, color='magenta', linestyle='--', linewidth = 1)
#     # ------------------------------------------------------------------------------------>
#     plt.subplot(412)
#
#     plt.xlim(-10, samp_len + 10)
#     plt.ylim(miny, maxy)
#
#     plt.xlabel('Samples [4ns]', fontsize=10)
#     plt.ylabel('ch1 [V]', fontsize=10)
#     plt.grid(True)
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.plot(sams, channel[1], color='black', linestyle='-', linewidth=.5, label='')
#     # plt.axvline(6500, color='magenta', linestyle='--', linewidth = 1)
#     # ------------------------------------------------------------------------------------>
#
#     plt.subplot(413)
#
#     plt.xlim(-10, samp_len + 10)
#     plt.ylim(miny, maxy)
#
#     plt.xlabel('Samples [4ns]', fontsize=10)
#     plt.ylabel('ch2 [V]', fontsize=10)
#     plt.grid(True)
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.plot(sams, channel[2], color='black', linestyle='-', linewidth=.5, label='')
#     # plt.axvline(6500, color='magenta', linestyle='--', linewidth = 1)
#     # ------------------------------------------------------------------------------------>
#     plt.subplot(414)
#
#     plt.xlim(-10, samp_len + 10)
#     plt.ylim(miny, maxy)
#
#     plt.xlabel('Samples [4ns]', fontsize=10)
#     plt.ylabel('ch3 [V]', fontsize=10)
#     plt.grid(True)
#     plt.xticks(fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.plot(sams, channel[3], color='black', linestyle='-', linewidth=.5, label='')
#     # plt.axvline(6500, color='magenta', linestyle='--', linewidth = 1)
#     # ------------------------------------------------------------------------------------>
#     ######################################################################################################:
#     ## Summed WF:
#     fig, ax = plt.subplots(figsize=(15, 7))
#     plt.xlim(0, samp_len)
#     ## plt.xlim(4900,  5100)
#     ## plt.ylim(0,  0.5)
#     plt.xlabel('Samples', fontsize=15)
#     plt.ylabel('Amp [V]', fontsize=15)
#     plt.grid(True)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.plot(sams, dat_raw_temp, color='black', linestyle='-', linewidth=0.5, label='Summed Waveform')
#     plt.plot(sams, data_smooth, color='springgreen', linestyle='-', linewidth=1., alpha=1, label='Smoothed Waveform')
#
#     ## plot baseline:
#     plt.axhline(s1_thre_base * BaseLineSumSigma[0], color='magenta', linestyle='--', linewidth=1., label='S1 Threshold')
#     plt.axhline(s2_thre_base * BaseLineSumSigma[0], color='yellow', linestyle='--', linewidth=1., label='S2 Threshold')
#     ## S1:
#     if len(S1) > 0:
#         ## Marker all S1s ===>
#         for p1 in range(len(S1)):
#             if p1 == 0:
#                 ax.axvspan(S1[p1][0], S1[p1][1], alpha=0.4, color='magenta', label='S1')
#             else:
#                 ax.axvspan(S1[p1][0], S1[p1][1], alpha=0.4, color='magenta', label='')
#                 ## Marker the Largest S1 ===>
#         plt.text(S1sPeak[0] + 200, data_smooth[int(S1sPeak[0])], 'S1[0]\n\n%.1f PE' % (S1sTot[0]), fontsize=10,
#                  color='blue', fontweight='bold', zorder=100)
#
#     ## S2:
#     if len(S2) > 0:
#         ## Marker all S2s
#         for p2 in range(len(S2)):
#             if p2 == 0:
#                 # ax.axvspan(peaklow[p2][0], peaklow[p2][2], alpha=0.4, facecolor='magenta', edgecolor = 'black', label = 'S2')
#                 ax.axvspan(S2[p2][0], S2[p2][1], alpha=0.4, facecolor='yellow', edgecolor='black', label='S2')
#                 # print ('========> ',peaklow[p2][0], peaklow[p2][2])
#             else:
#                 ax.axvspan(S2[p2][0], S2[p2][1], alpha=0.4, facecolor='yellow', edgecolor='black', label='')
#                 # print ('========> ',peaklow[p2][0], peaklow[p2][2])
#         ## Marker the Largest S2 ===>
#         plt.text(S2sPeak[0] + 200, data_smooth[int(S2sPeak[0])], 'S2[0]\n\n%.1f PE' % (S2sTot[0]), fontsize=10,
#                  color='blue', fontweight='bold', zorder=100)
#
#     ## Test:
#     ## print 'Max smooth: ',np.max(data_smooth)
#
#     plt.title(fname[-19:-4] + ' event ' + str(evt))
#     plt.legend(loc='best', shadow=True, fontsize=12.5)
#     plt.ylim(0, )
#     if savepath:
#         name = fname[-26:-4] + '_event_' + str(evt) + '.png'
#         plt.savefig(os.path.join(savepath, name), dpi=300)
#         plt.close()
#     else:
#         plt.show(block=False)
#         raw_input("Hit Enter To Close")
#         plt.close()

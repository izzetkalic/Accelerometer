# Necessary imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
from pandas import DataFrame as df
import glob

style.use('ggplot')


# Either we can import one by one or all of them of csv files. I chose to import all of them.
def getfiles(path):
    # Getting file names
    fileNamesWExt = os.listdir(path)
    listNames = []
    for fileName in fileNamesWExt:
        listNames.append(os.path.splitext(fileName))
    names = []
    for name in listNames:
        names.append(name[0])
    # Getting files to data frame
    filePaths = glob.glob(path + "/*.csv")
    fileList = []
    for path in filePaths:
        dataFrame = df.from_csv(path)
        dataFrame.columns = ['z', 'y', 'x']
        fileList.append(dataFrame)
    return names, fileList


# Week 1 10x10 Full Range
week1FullPath = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_1_10x10\Full'  # with 'r' IDE recognise the
# sentence in quotes as a string
names, fileList = getfiles(week1FullPath)
# Assigning all data frame to their file name
for i in range(0, 9):
    locals()[names[i]] = fileList[i]

# Week 1 10x10 Partial Range
week1PartialPath = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_1_10x10\Partial'
names, fileList = getfiles(week1PartialPath)
for i in range(0, 9):
    locals()[names[i]] = fileList[i]

# Week 2 5x10 Full Range
week2FullPath = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_2_5x10\Full'
names, fileList = getfiles(week2FullPath)
for i in range(0, 9):
    locals()[names[i]] = fileList[i]

# Week 2 5x10 Partial Range
week2PartialPath = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_2_5x10\Partial'
names, fileList = getfiles(week2PartialPath)
for i in range(0, 9):
    locals()[names[i]] = fileList[i]

localVariables = locals()

# ALL CSV FILES Blue=X-Axis Green=Y-Axis Red=Z-Axis (Data Structure: date/time/z/y/x)
blue_patch = mpatches.Patch(color='blue', label='X-Axis')
green_patch = mpatches.Patch(color='green', label='Y-Axis')
red_patch = mpatches.Patch(color='red', label='Z-Axis')

# For Left Hand Data
plt.figure(1)
plt.subplot(411)  # 1 means: 1st Graph 41 means: 4x1 matrix grid
plt.plot(localVariables['fbenchL1week'])
plt.title('Full Range Bench LEFT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

plt.subplot(412)
plt.plot(localVariables['pbenchL1week'])
plt.title('Partial Range Bench LEFT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

# For Right Hand Data
plt.subplot(413)
plt.plot(localVariables['fbenchR1week'])
plt.title('Full Range Bench RIGHT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

plt.subplot(414)
plt.plot(localVariables['pbenchR1week'])
plt.title('Partial Range Bench RIGHT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')
plt.show()

from scipy import signal

# Filtering
fullRange = localVariables['fbenchL1week']['2016-07-04 11:32:35.000':'2016-07-04 11:32:51.000']
b, a = signal.butter(8, 1 / 50, 'high')
output_signal = signal.filtfilt(b, a, fullRange.y)
partialRange = localVariables['pbenchL1week']['2016-07-09 12:54:22.000':'2016-07-09 12:54:39.000']
b, a = signal.butter(3, 1 / 50, 'high')
output_signal2 = signal.filtfilt(b, a, partialRange.y)
plt.figure(3)
plt.subplot(411)
plt.plot(fullRange.y)
plt.subplot(412)
plt.plot(output_signal)
plt.subplot(413)
plt.plot(partialRange.y)
plt.subplot(414)
plt.plot(output_signal2)

# For FFT
plt.figure(4)
plt.subplot(221)
fullRange = localVariables['fbenchL1week']['2016-07-04 11:32:35.000':'2016-07-04 11:32:51.000']
plt.title('Full Range Bench LEFT Hand')
plt.plot(fullRange)

plt.subplot(222)
partialRange = localVariables['pbenchL1week']['2016-07-09 12:54:22.000':'2016-07-09 12:54:39.000']
plt.title('Partial Range Bench LEFT Hand')
plt.plot(partialRange)

plt.subplot(223)
plt.title('Full Range Bench LEFT Hand FFT')
A = np.fft.fft(fullRange.y)
plt.plot(np.abs(A) ** 2)

plt.subplot(224)
plt.title('Partial Range Bench LEFT Hand FFT')
plt.plot(np.fft.fft(partialRange.y))

# FFT with filter
plt.figure(5)
plt.subplot(211)
plt.plot(output_signal)
plt.subplot(212)
plt.plot(np.fft.fft(output_signal))

import numpy as np
from scipy import signal
from peakdetect import peakdetect
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import style
from pandas import DataFrame as df

style.use('ggplot')


def get_periodic(path):
    periodics = []
    data_frame = df.from_csv(path)
    data_frame.columns = ['z', 'y', 'x']
    if path.__contains__('1'):
        if path.__contains__('bench'):
            bench_press_1_week = data_frame.between_time('11:24:30', '11:51:45')
            peak_indexes = get_peaks(bench_press_1_week.y, lookahead=4000)
            lag = 0
            for i in range(0, len(peak_indexes)):
                time_indexes = bench_press_1_week.index.tolist()
                start_time = time_indexes[0]
                periodic_start = start_time.to_datetime() + dt.timedelta(seconds=(peak_indexes[i] / 100) - lag)
                lag = lag + 3
                periodic_end = periodic_start + dt.timedelta(seconds=30)
                periodic = bench_press_1_week.between_time(periodic_start.time(), periodic_end.time())
                periodics.append(periodic)
    return periodics


def get_peaks(data, lookahead):
    correlation = signal.correlate(data, data, mode='full')
    realcorr = correlation[correlation.size / 2:]
    maxpeaks, minpeaks = peakdetect(realcorr, lookahead=lookahead)
    x, y = zip(*maxpeaks)

    return x


def show_segment_plot(data, periodic_area, exercise_name):
    plt.figure(8)
    gs = gridspec.GridSpec(7, 2)
    ax = plt.subplot(gs[:2, :])
    plt.title(exercise_name)
    ax.plot(data)
    k = 0
    for i in range(2, 7):
        for j in range(0, 2):
            ax = plt.subplot(gs[i, j])
            title = "{} {}".format(k + 1, ".Set")
            plt.title(title)
            ax.plot(periodic_area[k])
            k = k + 1
    plt.show()


bench_press_1_week = fbenchL1week.between_time('11:24:30', '11:51:45')
correlation = signal.correlate(bench_press_1_week.y, bench_press_1_week.y, mode='full')
realcorr = correlation[correlation.size / 2:]
maxpeaks, minpeaks = peakdetect(realcorr, lookahead=10000)

signal.correlate(bench_press_1_week.y, bench_press_1_week.y, mode='full')

periodic_area = get_periodic(r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_1_10x10\Full\fbenchL1week.csv')


path = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_1_10x10\Partial\fbenchL1week.csv'
import re
re.findall(r'\\(.*?).csv',path)
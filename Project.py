# Necessary imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pandas import DataFrame as df
import glob

# Either we can import one by one or all of them of csv files. I chose to import all of them.
def getFiles(path):
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
        fileList.append(dataFrame)
    return names, fileList

# Week 1 10x10 Full Range
week1FullPath = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_1_10x10\Full'  # with 'r' IDE recognise the
# sentence in quotes as a string
names, fileList = getFiles(week1FullPath)
# Assigning all data frame to their file name
for i in range(0, 9):
    locals()[names[i]] = fileList[i]

# Week 1 10x10 Partial Range
week1PartialPath = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_1_10x10\Partial'
names, fileList = getFiles(week1PartialPath)
for i in range(0, 9):
    locals()[names[i]] = fileList[i]

# Week 2 5x10 Full Range
week2FullPath = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_2_5x10\Full'
names, fileList = getFiles(week2FullPath)
for i in range(0, 9):
    locals()[names[i]] = fileList[i]

# Week 2 5x10 Partial Range
week2PartialPath = r'C:\Users\Raven\PycharmProjects\Accelerometer\PA1\Week_2_5x10\Partial'
names, fileList = getFiles(week2PartialPath)
for i in range(0, 9):
    locals()[names[i]] = fileList[i]

# ALL CSV FILES Blue=X-Axis Green=Y-Axis Red=Z-Axis (Data Structure: date/time/z/y/x)
blue_patch = mpatches.Patch(color='blue', label='X-Axis')
green_patch = mpatches.Patch(color='green', label='Y-Axis')
red_patch = mpatches.Patch(color='red', label='Z-Axis')

# For Left Hand Data
plt.figure(1)
plt.subplot(411)  # 1 means: 1st Graph 41 means: 4x1 matrix grid
plt.plot(fbenchL1week)
plt.title('Full Range Bench LEFT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

plt.subplot(412)
plt.plot(pbenchL1week)
plt.title('Partial Range Bench LEFT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

# For Right Hand Data
plt.subplot(413)
plt.plot(fbenchR1week)
plt.title('Full Range Bench RIGHT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

plt.subplot(414)
plt.plot(pbenchR1week)
plt.title('Partial Range Bench RIGHT Hand')
plt.legend(handles=[blue_patch,green_patch,red_patch],fontsize='x-small')
plt.show()
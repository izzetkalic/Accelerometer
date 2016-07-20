# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Importing data
my_data = np.genfromtxt('PA1\\Week_1_10x10\\04072016fbenchpcurlL.csv', delimiter=',')
axises=my_data[ : , 1:4]

# Plot
plt.plot(axises)

# Histogram
plt.hist(axises)

# ALL CSV FILES Blue=X-Axis Green=Y-Axis Red=Z-Axis (Data Structure: date/time/z/y/x)
#  Week 1 10x10
f_bench_l = np.genfromtxt('PA1\\Week_1_10x10\\full\\04072016fbenchL.csv', delimiter=',',usecols=[1,2,3])
f_bench_r = np.genfromtxt('PA1\\Week_1_10x10\\full\\04072016fbenchR.csv', delimiter=',',usecols=[1,2,3])
p_bench_l = np.genfromtxt('PA1\\Week_1_10x10\\partial\\09072016pbenchL.csv', delimiter=',',usecols=[1,2,3])
p_bench_r = np.genfromtxt('PA1\\Week_1_10x10\\partial\\09072016pbenchR.csv', delimiter=',',usecols=[1,2,3])


blue_patch = mpatches.Patch(color='blue', label='X-Axis')
green_patch = mpatches.Patch(color='green', label='Y-Axis')
red_patch = mpatches.Patch(color='red', label='Z-Axis')
# For Left Hand Data
plt.figure(1)
plt.subplot(411)  # 1 means: 1st Graph 41 means: 4x1 matrix grid
plt.plot(f_bench_l)
plt.title('Full Range Bench LEFT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

plt.subplot(412)
plt.plot(p_bench_l)
plt.title('Partial Range Bench LEFT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

# For Right Hand Data
plt.subplot(413)
plt.plot(f_bench_r)
plt.title('Full Range Bench RIGHT Hand')
plt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')

plt.subplot(414)
plt.plot(p_bench_r)
plt.title('Partial Range Bench RIGHT Hand')
plt.legend(handles=[blue_patch,green_patch,red_patch],fontsize='x-small')
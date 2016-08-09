import datetime as dt
import os
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import time
import numpy as np

import matplotlib as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as pyplt
from matplotlib import style
from pandas import DataFrame
from scipy import signal

from peakdetect import peakdetect

plt.use("TkAgg")
style.use('ggplot')

LARGE_FONT = ("Verdana", 12)
NORMAL_FONT = ("Verdana", 8)
SMALL_FONT = ("Verdana", 6)
blue_patch = mpatches.Patch(color='#951732', label='X-Axis')
green_patch = mpatches.Patch(color='#0b4545', label='Y-Axis')
red_patch = mpatches.Patch(color='#50AC3A', label='Z-Axis')


class ProjectApp(tk.Tk):  # The object inside the bracket is basically inherits the tkinter objects

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.file_paths = ["", ""]
        self.file_names = ["", ""]
        self.data_frames = [None, None, False, False]
        self.data_frames_for_analysis = [None, None, False, False]
        self.data_frame_segments = [[], [], False, False]

        tk.Tk.wm_title(self, "Accelerometer Analyzer")

        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, AnalysisPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

    def show_message(self, message):

        popup = tk.Tk()

        def close_mini():
            popup.destroy()

        popup.wm_title("!")
        label = ttk.Label(popup, text=message)
        label.pack(side='top', fill='x', pady=10, padx=10)
        btn_done = ttk.Button(popup, text="Okay!", command=close_mini)
        btn_done.pack()
        popup.mainloop()

    def get_file(self, is_first, label):

        file_path = askopenfilename()

        if file_path:
            if is_first:
                self.file_paths[0] = file_path
                self.file_names[0] = os.path.basename(file_path).partition('.')[0]
                self.data_frames[0] = DataFrame.from_csv(self.file_paths[0])
                self.data_frames[2] = True
                self.data_frames[0].columns = ['x', 'y', 'z']
            else:
                self.file_paths[1] = file_path
                self.file_names[1] = os.path.basename(file_path).partition('.')[0]
                self.data_frames[1] = DataFrame.from_csv(self.file_paths[1])
                self.data_frames[3] = True
                self.data_frames[1].columns = ['x', 'y', 'z']
            label.config(text="Chosen")

    def show_filter(self, is_first):
        if is_first:
            which_file = 0
        else:
            which_file = 1

        b, a = signal.butter(4, 0.03, btype='lowpass')
        filtered_data = signal.filtfilt(b, a, self.data_frames[which_file].y)
        pyplt.figure(which_file + 1)
        pyplt.plot(filtered_data)
        pyplt.show()

    def start_analysis(self, label_waiting, start_time_1, end_time_1, start_time_2,
                       end_time_2, set_time_1, set_time_2):

        def write_times_to_file(start_1, end_1, start_2, end_2, set_time_1, set_time_2):
            file = open("remember.txt", "w")
            file.write(start_1 + "\n")
            file.write(end_1 + "\n")
            file.write(start_2 + "\n")
            file.write(end_2 + "\n")
            file.write(str(set_time_1) + "\n")
            file.write(str(set_time_2) + "\n")
            file.close()

        def get_periodics(data_frame, which_file, set_time, limit, lag_time):

            def get_cut_points(data, limit):
                cut_points = []
                for i in range(0, 10):
                    if limit < 0:
                        start_index = np.where(data < limit)[0][0]
                    else:
                        start_index = np.where(data > limit)[0][0]
                    if i == 0:
                        cut_points.append(start_index)
                    else:
                        cut_points.append(cut_points[i - 1] + set_time * 100 + start_index)
                    cut_point = start_index + set_time * 100
                    data = data[cut_point::]

                return cut_points

            b, a = signal.butter(4, 0.03, btype='lowpass')
            filtered_data = signal.filtfilt(b, a, data_frame)

            cond = False
            if cond:
                pyplt.figure(which_file)
                pyplt.plot(filtered_data)
                pyplt.show()

            time_indexes = self.data_frames_for_analysis[which_file].index.tolist()
            start_time = time_indexes[0]
            data_frame = self.data_frames_for_analysis[which_file]
            cut_points = get_cut_points(filtered_data, limit)
            lag = 0
            for i in range(0, 10):
                periodic_start = start_time.to_datetime() + dt.timedelta(seconds=(cut_points[i] - 300) / 100 - lag)
                periodic_end = periodic_start + dt.timedelta(seconds=set_time + 5)
                lag += lag_time
                periodic = data_frame.between_time(periodic_start.time(), periodic_end.time())
                self.data_frame_segments[which_file].append(periodic)
            """
            for i in range(0, 10):
                start_index = np.where(filtered_data < -1.25)[0][0]
                periodic_start = start_time.to_datetime() + dt.timedelta(seconds=(start_index - lag) / 100)
                periodic_end = periodic_start + dt.timedelta(seconds=set_time)
                periodic = data_frame.between_time(periodic_start.time(), periodic_end.time())
                cut_point = start_index + set_time * 100
                filtered_data = filtered_data[cut_point::]
                cut_time = start_time.to_datetime() + dt.timedelta(seconds=cut_point / 100)
                data_frame = data_frame.between_time(cut_time.time(), end_1)
                start_time = data_frame.index.tolist()[0]
                self.data_frame_segments[which_file].append(periodic)
                lag += 300
            """

        def segment_data(file_name, is_first, set_time):

            if is_first:
                which_file = 0
            else:
                which_file = 1

            self.data_frame_segments[which_file + 2] = True

            if file_name.__contains__('bench'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=-1.25,
                                      lag_time=3)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=-1.25,
                                      lag_time=0)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=-1.25,
                                      lag_time=0.5)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=-1.25,
                                      lag_time=0)
            elif file_name.__contains__('military'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
            elif file_name.__contains__('curl'):
                # TODO bak buna bu nedir abi
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0.95,
                                      lag_time=1.5)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0.95,
                                      lag_time=2)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
            elif file_name.__contains__('deadlift'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
            elif file_name.__contains__('squat'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, limit=0,
                                      lag_time=0)

        if not self.data_frames[2]:
            self.show_message('Please choose first file')
        elif not self.data_frames[3]:
            self.show_message('Please choose second file')
        else:
            if start_time_1.get() == "":
                self.show_message('Please Fill 1. Time Area (eg. 00:00:00)')
            elif end_time_1.get() == "":
                self.show_message('Please Fill 2. Time Area (eg. 00:00:00)')
            elif start_time_2.get() == "":
                self.show_message('Please Fill 3. Time Area (eg. 00:00:00)')
            elif end_time_2.get() == "":
                self.show_message('Please Fill 4. Time Area (eg. 00:00:00)')
            elif set_time_1.get() == "":
                self.show_message('Plese Fill 1. Set Time (in second)')
            elif set_time_2.get() == "":
                self.show_message('Plese Fill 2. Set Time (in second)')
            else:
                start_1 = str(start_time_1.get())
                end_1 = str(end_time_1.get())
                start_2 = str(start_time_2.get())
                end_2 = str(end_time_2.get())
                set_time_1 = int(set_time_1.get())
                set_time_2 = int(set_time_2.get())
                write_times_to_file(start_1, end_1, start_2, end_2, set_time_1, set_time_2)
                self.data_frames_for_analysis[0] = self.data_frames[0].between_time(start_1, end_1)
                self.data_frames_for_analysis[2] = True
                self.data_frames_for_analysis[1] = self.data_frames[1].between_time(start_2, end_2)
                self.data_frames_for_analysis[3] = True
                label_waiting.config(text="First file is analysing")
                segment_data(self.file_names[0], True, set_time_1)
                label_waiting.config(text="Second file is analysing")
                segment_data(self.file_names[1], False, set_time_2)
                self.show_frame(AnalysisPage)

    def new_window(self):

        window = tk.Toplevel(self)
        label = tk.Label(window)
        label.pack(side="top", fill="both", padx=10, pady=10)

    def show_plot(self, is_first, data_frames):

        if is_first:
            if pyplt.fignum_exists(1):
                self.show_message('Figure 1 is already showing')
            else:
                title = self.create_title(self.file_names[0])
                self.create_plot(data_frames[0], 1, title)
        else:
            if pyplt.fignum_exists(2):
                self.show_message('Figure 2 is already showing')
            else:
                title = self.create_title(self.file_names[1])
                self.create_plot(data_frames[1], 2, title)

    def create_plot(self, data_frame, figure, title):

        pyplt.figure(figure)
        pyplt.plot(data_frame['x'], color='#951732')
        pyplt.plot(data_frame['y'], color='#0b4545')
        pyplt.plot(data_frame['z'], color='#50AC3A')
        pyplt.xlabel('Time')
        pyplt.ylabel('Acceleration')
        pyplt.title(title)
        pyplt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')
        pyplt.show()

    def create_title(self, file_name):

        content = []
        if file_name[0] == 'f':
            content.append('Full Range')
        else:
            content.append('Partial Range')
        if file_name.__contains__('bench'):
            content.append('Bench Press')
        elif file_name.__contains__('military'):
            content.append('Military Press')
        elif file_name.__contains__('curl'):
            content.append('Barbell Curl')
        elif file_name.__contains__('deadlift'):
            content.append('Deadlift')
        elif file_name.__contains__('squat'):
            content.append('Squat')
        if file_name.__contains__('L'):
            content.append('Left Hand')
        else:
            content.append('Right Hand')
        if file_name.__contains__('1'):
            content.append('1 Week Data')
        elif file_name.__contains__('w'):
            content.append('2 Week Data')
        else:
            content.append('Data')
        title = "{} {} {} {}".format(content[0], content[1], content[2], content[3])

        return title

    def get_periodic(self):
        print('Merhaba')

    def get_periodic_old(self, data_frame, lookahead, lag_time, set_time, which_file):

        def get_peaks(data, lookahead):
            correlation = signal.correlate(data, data, mode='full')
            realcorr = correlation[correlation.size / 2:]
            maxpeaks, minpeaks = peakdetect(realcorr, lookahead=lookahead)
            x, y = zip(*maxpeaks)

            return x

        peak_indexes = get_peaks(data_frame.y, lookahead=lookahead)
        lag = 0
        for i in range(0, len(peak_indexes)):
            time_indexes = data_frame.index.tolist()
            start_time = time_indexes[0]
            periodic_start = start_time.to_datetime() + dt.timedelta(seconds=(peak_indexes[i] / 100) - lag)
            lag += lag_time
            periodic_end = periodic_start + dt.timedelta(seconds=set_time)
            periodic = data_frame.between_time(periodic_start.time(), periodic_end.time())
            self.data_frame_segments[which_file].append(periodic)

    def show_sets(self, file_name, is_first):

        if is_first:
            figure = 3
            which_file = 0
        else:
            figure = 4
            which_file = 1

        if is_first:
            if pyplt.fignum_exists(3):
                self.show_message('Figure 3 is already showing')
            else:
                title = self.create_title(file_name)
                self.create_segment_plot(self.data_frames_for_analysis[which_file], figure, title, which_file=0)
        else:
            if pyplt.fignum_exists(4):
                self.show_message('Figure 4 is already showing')
            else:
                title = self.create_title(file_name)
                self.create_segment_plot(self.data_frames_for_analysis[which_file], figure, title, which_file=1)

    def create_segment_plot(self, data_frame, figure, title, which_file):
        pyplt.figure(figure)
        gs = gridspec.GridSpec(7, 2)
        segment_plot = pyplt.subplot(gs[:2, :])
        pyplt.title(title)
        pyplt.xlabel('Time')
        pyplt.ylabel('Acceleration')
        pyplt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')
        segment_plot.plot(data_frame['x'], color='#951732')
        segment_plot.plot(data_frame['y'], color='#0b4545')
        segment_plot.plot(data_frame['z'], color='#50AC3A')
        k = 0
        for i in range(2, 7):
            for j in range(0, 2):
                segment_plot = pyplt.subplot(gs[i, j])
                segment_title = "{} {}".format("Set", k + 1)
                pyplt.title(segment_title)
                pyplt.xlabel('Time')
                pyplt.ylabel('Acceleration')
                pyplt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')
                segment_plot.plot(self.data_frame_segments[which_file][k]['x'], color='#951732')
                segment_plot.plot(self.data_frame_segments[which_file][k]['y'], color='#0b4545')
                segment_plot.plot(self.data_frame_segments[which_file][k]['z'], color='#50AC3A')
                k += 1
        pyplt.show()


class StartPage(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.controller = controller
        self.parent = parent

        outer_frame = tk.Frame(self)
        outer_frame.pack(padx=50, pady=50)

        inner_frame = tk.Frame(outer_frame)
        inner_frame.grid(row=0, column=0, columnspan=2, pady=10)

        label_greeting = tk.Label(inner_frame, text="Please Choose Files", font=LARGE_FONT)
        label_greeting.grid(row=0, column=0, columnspan=4, sticky='ew')

        btn_first_file = ttk.Button(inner_frame, text="First File",
                                    command=lambda: controller.get_file(True, label_first_file))
        btn_first_file.grid(row=1, column=0)

        label_first_file = tk.Label(inner_frame, text="Waiting")
        label_first_file.grid(row=1, column=1)

        btn_plot_first = ttk.Button(inner_frame, text="Plot First File",
                                    command=lambda: controller.show_plot(True, controller.data_frames))
        btn_plot_first.grid(row=1, column=2, sticky='ew', padx=15)

        btn_show_filter_first = ttk.Button(inner_frame, text="Show First Filtered",
                                           command=lambda: controller.show_filter(True))
        btn_show_filter_first.grid(row=1, column=3, sticky='ew')

        btn_second_file = ttk.Button(inner_frame, text="Second File",
                                     command=lambda: controller.get_file(False, label_second_file))
        btn_second_file.grid(row=2, column=0)

        label_second_file = tk.Label(inner_frame, text="Waiting")
        label_second_file.grid(row=2, column=1)

        btn_plot_second = ttk.Button(inner_frame, text="Plot Second File",
                                     command=lambda: controller.show_plot(False, controller.data_frames))
        btn_plot_second.grid(row=2, column=2, sticky='ew', padx=15)

        btn_show_filter_second = ttk.Button(inner_frame, text="Show Second Filtered",
                                            command=lambda: controller.show_filter(False))
        btn_show_filter_second.grid(row=2, column=3, sticky='ew')

        left_frame = tk.Frame(outer_frame)
        left_frame.grid(row=1, column=0, padx=10)

        label_first = ttk.Label(left_frame, text="For First Dataset")
        label_first.grid(row=0, column=0, columnspan=2, sticky='ew')

        file = open("remember.txt", "r")
        texts_for_entry = file.read().splitlines()

        label_start_first = ttk.Label(left_frame, text="Start Time:")
        label_start_first.grid(row=1, column=0, sticky='w')

        entry_start_first = ttk.Entry(left_frame)
        entry_start_first.grid(row=1, column=1, sticky='w')

        label_end_first = ttk.Label(left_frame, text="End Time:")
        label_end_first.grid(row=2, column=0, sticky='w')

        entry_end_first = ttk.Entry(left_frame)
        entry_end_first.grid(row=2, column=1, sticky='w')

        label_set_time_first = ttk.Label(left_frame, text='Set Time')
        label_set_time_first.grid(row=3, column=0, sticky='w')

        entry_set_time_first = ttk.Entry(left_frame)
        entry_set_time_first.grid(row=3, column=1, sticky='w')

        right_frame = tk.Frame(outer_frame)
        right_frame.grid(row=1, column=1, pady=10)

        label_second = ttk.Label(right_frame, text="For Second Dataset")
        label_second.grid(row=0, column=0, columnspan=2, sticky='ew')

        label_start_second = ttk.Label(right_frame, text="Start Time:")
        label_start_second.grid(row=1, column=0, sticky='w')

        entry_start_second = ttk.Entry(right_frame)
        entry_start_second.grid(row=1, column=1, sticky='w')

        label_end_second = ttk.Label(right_frame, text="End Time:")
        label_end_second.grid(row=2, column=0, sticky='w')

        entry_end_second = ttk.Entry(right_frame)
        entry_end_second.grid(row=2, column=1, sticky='w')

        label_set_time_second = ttk.Label(right_frame, text='Set Time')
        label_set_time_second.grid(row=3, column=0, sticky='w')

        entry_set_time_second = ttk.Entry(right_frame)
        entry_set_time_second.grid(row=3, column=1, sticky='w')

        if not os.stat("remember.txt").st_size == 0:
            entry_start_first.insert(0, texts_for_entry[0])
            entry_end_first.insert(0, texts_for_entry[1])
            entry_start_second.insert(0, texts_for_entry[2])
            entry_end_second.insert(0, texts_for_entry[3])
            entry_set_time_first.insert(0, texts_for_entry[4])
            entry_set_time_second.insert(0, texts_for_entry[5])

        file.close()

        label_waiting = ttk.Label(outer_frame)
        label_waiting.grid(row=3, column=0, columnspan=2)

        btn_start = ttk.Button(outer_frame, text="Start Analysis",
                               command=lambda: controller.start_analysis(label_waiting, entry_start_first,
                                                                         entry_end_first, entry_start_second,
                                                                         entry_end_second, entry_set_time_first,
                                                                         entry_set_time_second))
        btn_start.grid(row=2, column=0, columnspan=2)


class AnalysisPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.parent = parent

        outer_frame = tk.Frame(self)
        outer_frame.pack(padx=50, pady=50)

        label = ttk.Label(outer_frame, text="Analysis Page", font=LARGE_FONT)
        label.grid(row=0, column=0, pady=15)

        inner_frame = tk.Frame(outer_frame)
        inner_frame.grid(row=1, column=0)

        left_frame = tk.Frame(inner_frame)
        left_frame.grid(row=0, column=0, padx=10)

        right_frame = ttk.Frame(inner_frame)
        right_frame.grid(row=0, column=1, pady=5)

        ttk.Separator(inner_frame).grid(row=1, column=0, sticky="ew", columnspan=2, pady=5)

        bottom_frame = ttk.Frame(inner_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2)

        label_first = ttk.Label(left_frame, text="For First Dataset", font=LARGE_FONT)
        label_first.grid(row=0, column=0, columnspan=2, sticky='ew')

        btn_plot_data_first = ttk.Button(left_frame, text="Plot Data",
                                         command=lambda: controller.show_plot(True,
                                                                              controller.data_frames_for_analysis))
        btn_plot_data_first.grid(row=1, column=0, columnspan=2, sticky='ew')

        btn_show_sets_first = ttk.Button(left_frame, text="Show Sets",
                                         command=lambda: controller.show_sets(controller.file_names[0], True))
        btn_show_sets_first.grid(row=4, column=0, columnspan=2, sticky='ew')

        btn_statistics_first = ttk.Button(left_frame, text="Statistics")
        btn_statistics_first.grid(row=5, column=0, columnspan=2, sticky='ew')

        label_second = ttk.Label(right_frame, text="For Second Dataset", font=LARGE_FONT)
        label_second.grid(row=0, column=0, columnspan=2, sticky='ew')

        btn_plot_data_second = ttk.Button(right_frame, text="Plot Data",
                                          command=lambda: controller.show_plot(False,
                                                                               controller.data_frames_for_analysis))
        btn_plot_data_second.grid(row=1, column=0, columnspan=2, sticky='ew')

        btn_show_sets_second = ttk.Button(right_frame, text="Show Sets",
                                          command=lambda: controller.show_sets(controller.file_names[1], False))
        btn_show_sets_second.grid(row=4, column=0, columnspan=2, sticky='ew')

        btn_statistics_second = ttk.Button(right_frame, text="Statistics")
        btn_statistics_second.grid(row=5, column=0, columnspan=2, sticky='ew')

        btn_compare = ttk.Button(bottom_frame, text="Compare")
        btn_compare.grid(row=0, column=0)

        btn_back = ttk.Button(bottom_frame, text="Back Home", command=lambda: controller.show_frame(StartPage))
        btn_back.grid(row=0, column=1)


app = ProjectApp()
app.resizable(False, False)
app.mainloop()

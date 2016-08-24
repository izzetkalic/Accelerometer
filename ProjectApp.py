import csv
import datetime as dt
import os
import sys
import time
import tkinter as tk
import tkinter.font as font
from threading import Thread
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import warnings

import matplotlib as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as pyplt
import numpy as np
from matplotlib import style
from pandas import DataFrame
from scipy import signal
from sklearn import tree

from detect_peaks import detect_peaks

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
        self.program_path = os.getcwd()
        self.waiting = tk.PhotoImage(file="icons/waiting.png")
        self.graph = tk.PhotoImage(file="icons/graph.png")
        self.check = tk.PhotoImage(file="icons/check.png")
        self.error = tk.PhotoImage(file="icons/error.png")
        self.info = tk.PhotoImage(file="icons/info.png")
        self.edit = tk.PhotoImage(file="icons/edit.png")
        self.statistics = tk.PhotoImage(file="icons/statistics.png")
        self.segment = tk.PhotoImage(file="icons/segment.png")
        self.restart = tk.PhotoImage(file="icons/restart.png")
        self.compare = tk.PhotoImage(file="icons/compare.png")
        self.loader = tk.PhotoImage(file="icons/loader.gif")
        self.data_frames = [None, None, False, False]
        self.data_frames_for_analysis = [None, None]
        self.data_frame_segments = [[], []]
        self.data_frame_transformed = [[], []]
        self.data_frame_statistics = DataFrame()
        self.whole_file = []
        self.data_frame_parameters = [[], []]
        """
        which_file: indicates which file had choosen
        self.data_frame_parameters[which_file][0]: File Name
        self.data_frame_parameters[which_file][1]: Start Time of Execise
        self.data_frame_parameters[which_file][2]: End Time of Exercise
        self.data_frame_parameters[which_file][3]: Set Time of Exercise
        self.data_frame_parameters[which_file][4]: Rest Time of Exercise
        self.data_frame_parameters[which_file][5]: Treshold for Segmentation
        self.data_frame_parameters[which_file][6]: Lag Time
        self.data_frame_parameters[which_file][7]: User's Intend of Filtering
        self.data_frame_parameters[which_file][8]: Order of Filter
        self.data_frame_parameters[which_file][9]: Wn
        self.data_frame_parameters[which_file][10]: Butter Filter Type
        """
        self.process_counter = 0
        self.threads = []
        self.index_error = 0
        self.allow_thread_3 = True
        self.column_headers_for_statistics = ["Set", "Rep", "Hand", "EType", "Mx", "My", "Mz", "STDx", "STDy", "STDz",
                                              "FREQ1x", "POW1x", "FREQ1y", "POW1y", "FREQ1z", "POW1z"]

        tk.Tk.wm_title(self, "Accelerometer Analyzer")

        if not os.path.isfile(self.program_path + "/remember.csv"):
            with open('remember.csv', 'w'):
                pass

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

    def parameters(self, is_first, label):

        def close_protocol(window, frame):

            entries = []
            for child in frame.winfo_children():
                if type(child) is ttk.Entry:
                    entries.append(child)
            if not all(len(child.get()) > 0 for child in entries):
                self.show_message("Please fill all fields")
            else:
                window.destroy()

        def fill_parameter_for_file(which_file, frame, window, is_filter, button):

            entries = []

            for child in frame.winfo_children():
                if type(child) is ttk.Entry:
                    entries.append(child)

            if not all(len(child.get()) > 0 for child in entries):
                self.show_message("Please fill all fields")
            elif (is_filter.get() == 1) & (len(self.data_frame_parameters[which_file][10]) == 0):
                self.show_message("Plese fill filter parameters")
            else:
                self.data_frame_parameters[which_file][0] = self.file_names[which_file]
                for i, entry in enumerate(entries):
                    self.data_frame_parameters[which_file][i + 1] = entry.get()
                self.data_frame_parameters[which_file][7] = is_filter.get()

                button.config(image=self.check)
                window.destroy()

        def set_filter_parameter(frame, is_filter):

            def show_filter(order, cutoff, sampling_freq, type):

                nyq = 0.5 * float(sampling_freq.get())
                normal_cutoff = float(cutoff.get()) / nyq
                b, a = signal.butter(float(order.get()), normal_cutoff, btype=type.get())
                filtered_data = signal.filtfilt(b, a, self.data_frames[which_file].y)
                pyplt.figure(which_file + 1)
                pyplt.plot(filtered_data)
                pyplt.show()

            def set_parameters(frame, window):

                entries = []
                for child in frame.winfo_children():
                    if type(child) is ttk.Entry:
                        entries.append(child)
                if not all(len(child.get()) > 0 for child in entries):
                    self.show_message("Please fill all fields")
                else:
                    i = 8
                    for child in frame_filter.winfo_children():
                        if type(child) is ttk.Entry:
                            self.data_frame_parameters[which_file][i] = child.get()
                            i += 1
                    window.destroy()

            if is_filter.get() == 1:

                entries = []
                for child in frame.winfo_children():
                    if type(child) is ttk.Entry:
                        entries.append(child)
                if not all(len(child.get()) > 0 for child in entries):
                    self.show_message("Please fill all fields first")
                else:
                    window_filter = tk.Toplevel(self)
                    window_filter.resizable(False, False)
                    tk.Tk.wm_title(window_filter, "Filter Parameters")

                    frame_filter = ttk.Frame(window_filter)
                    frame_filter.grid_propagate()
                    frame_filter.grid(row=1, column=0, sticky='ew', padx=15, pady=15)

                    label_order = ttk.Label(frame_filter, text="Filter Order: ")
                    label_order.grid(row=1, column=0, padx=3)

                    entry_order = ttk.Entry(frame_filter)
                    entry_order.grid(row=1, column=1, padx=3, pady=3)

                    order_info = "The order of the filter"
                    btn_order_info = ttk.Button(frame_filter, image=self.info,
                                                command=lambda: self.show_message(order_info))
                    btn_order_info.grid(row=1, column=2)

                    label_cutoff = ttk.Label(frame_filter, text="Cutoff Frequency: ")
                    label_cutoff.grid(row=2, column=0, padx=3, sticky='w')

                    entry_cutoff = ttk.Entry(frame_filter)
                    entry_cutoff.grid(row=2, column=1, padx=3, pady=3)

                    cutoff_info = "Desired cutoff frequency of the filter(Hz)"
                    btn_cutoff_info = ttk.Button(frame_filter, image=self.info,
                                                 command=lambda: self.show_message(cutoff_info))
                    btn_cutoff_info.grid(row=2, column=2)

                    label_sampling_freq = ttk.Label(frame_filter, text="Sampling Frequency: ")
                    label_sampling_freq.grid(row=3, column=0, padx=3, sticky='w')

                    entry_sampling_freq = ttk.Entry(frame_filter)
                    entry_sampling_freq.grid(row=3, column=1, padx=3, pady=3)

                    sampling_freq_info = "Sampling frequency of the machine(Hz)"
                    btn_sampling_freq = ttk.Button(frame_filter, image=self.info,
                                                   command=lambda: self.show_message(sampling_freq_info))
                    btn_sampling_freq.grid(row=3, column=2)

                    label_butter_type = ttk.Label(frame_filter, text="Butter Type: ")
                    label_butter_type.grid(row=4, column=0, padx=3)

                    entry_butter_type = ttk.Entry(frame_filter)
                    entry_butter_type.grid(row=4, column=1, padx=3, pady=3)

                    butter_type_info = "The type of filter.  Default is 'lowpass'"
                    btn_butter_type_info = ttk.Button(frame_filter, image=self.info,
                                                      command=lambda: self.show_message(butter_type_info))
                    btn_butter_type_info.grid(row=4, column=2)

                    frame_button = ttk.Frame(window_filter)
                    frame_button.grid(row=2, column=0, pady=5)

                    btn_done_filter = ttk.Button(frame_button, text="Done",
                                                 command=lambda: set_parameters(frame_filter, window_filter))
                    btn_done_filter.grid(row=0, column=0, sticky='ns')

                    btn_show_filter = ttk.Button(frame_button, image=self.graph,
                                                 command=lambda: show_filter(entry_order, entry_cutoff,
                                                                             entry_sampling_freq, entry_butter_type))
                    btn_show_filter.grid(row=0, column=1)

                    window_filter.protocol('WM_DELETE_WINDOW', lambda: close_protocol(window_filter, frame_filter))

                    if self.data_frame_parameters[which_file][0] == self.file_names[which_file]:
                        i = 8
                        for child in frame_filter.winfo_children():
                            if type(child) is ttk.Entry:
                                child.insert(0, str(self.data_frame_parameters[which_file][i]))
                                i += 1

        if is_first:
            which_file = 0
        else:
            which_file = 1

        if len(self.data_frame_parameters[which_file]) == 0:
            self.data_frame_parameters[which_file] = ["", 0, 0, 0, 0, 0, 0, 0, 0, 0, ""]

        if not self.data_frames[which_file + 2]:
            if which_file == 0:
                text = "Please choose first file"
            else:
                text = "Please choose second file"
            self.show_message(text)
        else:
            window = tk.Toplevel(self)
            window.resizable(False, False)
            tk.Tk.wm_title(window, "Parameters")

            outer_frame = ttk.Frame(window)
            outer_frame.pack(padx=15, pady=15)

            top_frame = ttk.Frame(outer_frame)
            top_frame.grid(row=0)
            if is_first:
                label_top = ttk.Label(top_frame, text="For First File")
            else:
                label_top = ttk.Label(top_frame, text="For Second File")
            label_top.pack()

            param_frame = ttk.Frame(outer_frame)
            param_frame.grid(row=1)

            label_start = ttk.Label(param_frame, text="Start Time:")
            label_start.grid(row=0, column=0, sticky='w')

            entry_start = ttk.Entry(param_frame)
            entry_start.grid(row=0, column=1, sticky='w', padx=3, pady=3)
            btn_plot = ttk.Button(param_frame, image=self.graph,
                                  command=lambda: self.show_plot(is_first, self.data_frames))
            btn_plot.grid(row=0, column=2, sticky='w', padx=3)

            label_end = ttk.Label(param_frame, text="End Time:")
            label_end.grid(row=1, column=0, sticky='w')

            entry_end = ttk.Entry(param_frame)
            entry_end.grid(row=1, column=1, sticky='w', padx=3, pady=3)

            label_set_time = ttk.Label(param_frame, text='Set Time:')
            label_set_time.grid(row=2, column=0, sticky='w')

            entry_set_time = ttk.Entry(param_frame)
            entry_set_time.grid(row=2, column=1, sticky='w', padx=3, pady=3)

            label_rest_time = ttk.Label(param_frame, text='Rest Time:')
            label_rest_time.grid(row=3, column=0, sticky='w')

            entry_rest_time = ttk.Entry(param_frame)
            entry_rest_time.grid(row=3, column=1, sticky='w', padx=3, pady=3)

            label_treshold = ttk.Label(param_frame, text='Treshold:')
            label_treshold.grid(row=4, column=0, sticky='w')

            entry_treshold = ttk.Entry(param_frame)
            entry_treshold.grid(row=4, column=1, sticky='w', padx=3, pady=3)

            treshold_info = "Average acceleration value of each set of exercise"
            btn_treshold_info = ttk.Button(param_frame, image=self.info,
                                           command=lambda: self.show_message(treshold_info))
            btn_treshold_info.grid(row=4, column=2, sticky='w', padx=3)

            label_lag_time = ttk.Label(param_frame, text='Lag Time:')
            label_lag_time.grid(row=5, column=0, sticky='w')

            entry_lag_time = ttk.Entry(param_frame)
            entry_lag_time.grid(row=5, column=1, sticky='w', padx=3, pady=3)

            lag_time_info = """
            Segmentation is being made by frequency domain. When the frequency
            has been transformed to time there could be shift in time. To prevent this
            specify a lag time if there is shift in time
            """
            btn_lag_time_info = ttk.Button(param_frame, image=self.info,
                                           command=lambda: self.show_message(lag_time_info))
            btn_lag_time_info.grid(row=5, column=2, sticky='w', padx=3)

            is_filter = tk.IntVar()
            check_filter = ttk.Checkbutton(param_frame, state=tk.ACTIVE, variable=is_filter, onvalue=1, offvalue=0)
            check_filter.grid(row=6, column=0)

            label_filter = ttk.Label(param_frame, text="Would you like to filter?")
            label_filter.grid(row=6, column=1)

            btn_filter = ttk.Button(param_frame, image=self.edit)
            btn_filter.config(command=lambda: set_filter_parameter(param_frame, is_filter))
            btn_filter.grid(row=6, column=2)

            bottom_frame = ttk.Frame(outer_frame)
            bottom_frame.grid(row=3, pady=3)

            btn_done = ttk.Button(bottom_frame, text="Done",
                                  command=lambda: fill_parameter_for_file(which_file, param_frame, window, is_filter,
                                                                          label))
            btn_done.pack()

            window.protocol('WM_DELETE_WINDOW', lambda: close_protocol(window, param_frame))

            if self.data_frame_parameters[which_file][0] == self.file_names[which_file]:
                i = 1
                for child in param_frame.winfo_children():
                    if type(child) is ttk.Entry:
                        child.insert(0, str(self.data_frame_parameters[which_file][i]))
                        i += 1
                if len(self.data_frame_parameters[which_file]) > 7:
                    if int(self.data_frame_parameters[which_file][7]) == 1:
                        is_filter.set(1)

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
        btn_done = ttk.Button(popup, text="Ok!", command=close_mini)
        btn_done.pack()
        popup.mainloop()

    def get_file(self, is_first, label_file, label_param):

        if is_first:
            which_file = 0
        else:
            which_file = 1

        file_path = askopenfilename()

        if file_path:
            self.data_frame_parameters[which_file] = []
            self.whole_file.clear()
            self.file_paths[which_file] = file_path
            self.file_names[which_file] = os.path.basename(file_path).partition('.')[0]
            with open('remember.csv', newline='') as remember:
                reader = csv.reader(remember, delimiter=',')
                for row in reader:
                    if row:
                        self.whole_file.append(row)
                        if row[0] == self.file_names[which_file]:
                            self.data_frame_parameters[which_file] = row
            self.data_frames[which_file] = DataFrame.from_csv(self.file_paths[which_file])
            self.data_frames[which_file + 2] = True
            self.data_frames[which_file].columns = ['x', 'y', 'z']
            label_file.config(image=self.check)
            if len(self.data_frame_parameters[which_file]) == 0:
                label_param.config(image=self.edit)
            else:
                label_param.config(image=self.check)

    def restart_test(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def build_tree(self, tree, data_frame, which_file):

        for col in data_frame.columns:
            tree.heading(col, text=col.title())
            # adjust the column's width to the header string
            tree.column(col, width=font.Font().measure(col.title()))
        if which_file == 0:
            data = data_frame[:100]
        elif which_file == 1:
            data = data_frame[100:]
        else:
            data = data_frame

        for values in data.values:

            rounded = []
            for elem in values:
                if type(elem) is str:
                    rounded.insert(len(rounded), elem)
                else:
                    rounded.insert(len(rounded), round(elem, 4))

            tree.insert('', 'end', values=rounded)
            # adjust column's width if necessary to fit each value
            for ix, val in enumerate(rounded):
                col_w = font.Font().measure(str(val))
                if tree.column(data_frame.columns[ix], width=None) < col_w:
                    tree.column(data_frame.columns[ix], width=col_w)

    def start_analysis(self, label_waiting):

        def write_params_to_file():

            if (len(self.data_frame_parameters[0]) == 0) | (len(self.data_frame_parameters[1]) == 0):
                self.show_message("Please fill parameters")
            else:

                if len(self.whole_file) == 0:
                    self.whole_file.append(self.data_frame_parameters[0])
                    self.whole_file.append(self.data_frame_parameters[1])
                else:
                    for which_file in range(0, 2):

                        cond = [True for elem in self.whole_file if
                                elem[0] == self.data_frame_parameters[which_file][0]]

                        if not cond:
                            self.whole_file.append(self.data_frame_parameters[which_file])
                        else:
                            for i, elem in enumerate(self.whole_file):
                                if elem[0] == self.data_frame_parameters[which_file][0]:
                                    self.whole_file[i] = self.data_frame_parameters[which_file]

                with open('remember.csv', 'w') as remember:
                    writer = csv.writer(remember, delimiter=',')
                    for elem in self.whole_file:
                        writer.writerow(elem)
                remember.close()

        def segment_data(is_first):

            def filter_data(data, params):

                nyq = 0.5 * float(params[10])
                normal_cutoff = float(params[9]) / nyq
                b, a = signal.butter(float(params[8]), normal_cutoff, btype=params[11])
                filtered_data = signal.filtfilt(b, a, data)

                return filtered_data

            if is_first:
                which_file = 0
            else:
                which_file = 1

            params = self.data_frame_parameters[which_file]
            data_frame = self.data_frames_for_analysis[which_file].y  # Major axis is y

            def get_cut_points(data, limit, set_time, rest_time):
                cut_points = []
                data = np.array(data)
                for i in range(0, 10):

                    if not len(data) == 0:
                        if limit < 0:
                            start_index = np.where(data < limit)[0][0]
                        else:
                            start_index = np.where(data > limit)[0][0]

                        if i == 0:
                            cut_points.append(start_index)
                        else:
                            cut_points.append(cut_points[i - 1] + set_time * 100 + rest_time * 100 + start_index)

                        cut_point = start_index + set_time * 100 + rest_time * 100

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            data = np.split(data, [cut_point])[1]
                return cut_points

            if int(params[7]) == 1:
                filtered_data = filter_data(data_frame, params)
            else:
                filtered_data = np.array(data_frame)

            time_indexes = self.data_frames_for_analysis[which_file].index.tolist()
            start_time = time_indexes[0]
            data_frame = self.data_frames_for_analysis[which_file]
            cut_points = get_cut_points(filtered_data, float(params[5]), float(params[3]), float(params[4]))

            try:
                lag = 0
                for i in range(0, 10):
                    periodic_start = start_time.to_datetime() + dt.timedelta(seconds=(cut_points[i] - 200) / 100 - lag)
                    periodic_end = periodic_start + dt.timedelta(seconds=float(params[3]))
                    periodic = data_frame.between_time(periodic_start.time(), periodic_end.time())
                    self.data_frame_segments[which_file].append(periodic)
                    lag += float(params[6])
            except IndexError:
                self.index_error = 1

            self.process_counter += 1

            if self.process_counter == 2:
                if self.index_error == 1:
                    self.show_message("Parameter Error: Please control parameters")
                else:
                    self.threads[3].start()

        def calculate_statistics():

            def calculate_fft(data_frame):

                def calculate_fft_for_each_axis(data, axis, mph):

                    if not len(data[axis]) == 0:
                        fft = np.abs(np.fft.fft(data[axis])) ** 2

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fft_real = fft[:fft.size / 2]

                    self.data_frame_transformed[which_file].append(fft_real)

                    # TODO mph ve threshold a bak bakalım (, threshold=50)
                    fft_peaks = detect_peaks(fft_real, mph=mph)
                    if not len(fft_peaks) == 0:
                        frequency = fft_peaks[0]
                        amplitude = fft_real[fft_peaks[0]]
                    else:
                        frequency = 0
                        amplitude = 0

                    fft_features = [frequency, amplitude]

                    return fft_features

                # mph = minimum peak height
                fft_features_x = calculate_fft_for_each_axis(data_frame, 'x', mph=10)
                fft_features_y = calculate_fft_for_each_axis(data_frame, 'y', mph=10)
                fft_features_z = calculate_fft_for_each_axis(data_frame, 'z', mph=10)

                axis_fft_features = [fft_features_x, fft_features_y, fft_features_z]

                return axis_fft_features

            time.sleep(1)
            data = []
            for which_file in range(0, 2):
                for xi, segment in enumerate(self.data_frame_segments[which_file]):

                    partitions = []
                    partition_time = float(self.data_frame_parameters[which_file][3]) / 10
                    time_indexes = segment.index.tolist()
                    start_time = time_indexes[0]
                    end_time = start_time.to_datetime() + dt.timedelta(seconds=partition_time)
                    for i in range(0, 10):
                        part = segment.between_time(start_time.time(), end_time.time())
                        start_time = end_time
                        end_time = start_time + dt.timedelta(seconds=partition_time)
                        partitions.append(part)

                    for xj, rep in enumerate(partitions):

                        statistics = []
                        means = np.mean(rep)
                        stds = np.std(rep)
                        axis_fft_features = calculate_fft(rep)

                        statistics.insert(0, xi + 1)
                        statistics.insert(1, xj + 1)
                        if self.file_names[which_file].__contains__('L'):
                            statistics.insert(2, "Left")
                        elif self.file_names[which_file].__contains__('R'):
                            statistics.insert(2, "Right")
                        if self.file_names[which_file][0] == 'f':
                            statistics.insert(3, "Full Range")
                        else:
                            statistics.insert(3, "Partial Range")
                        statistics.extend(means)
                        statistics.extend(stds)
                        statistics.insert(10, axis_fft_features[0][0])
                        statistics.insert(11, axis_fft_features[0][1])
                        statistics.insert(12, axis_fft_features[1][0])
                        statistics.insert(13, axis_fft_features[1][1])
                        statistics.insert(14, axis_fft_features[2][0])
                        statistics.insert(15, axis_fft_features[2][1])

                        data.append(statistics)

            self.data_frame_statistics = DataFrame(data=data, columns=self.column_headers_for_statistics)

            self.show_frame(AnalysisPage)

        def animate():
            gif_index = 0
            while True:
                try:
                    time.sleep(0.04)

                    self.loader.config(format="gif - {}".format(gif_index))
                    label_waiting.config(text="Analysing Data")
                    label_waiting.config(image=self.loader)
                    label_waiting.config(compound="right")
                    label_waiting.image = self.loader

                    gif_index += 1

                except:
                    gif_index = 0

        if not self.data_frames[2]:
            self.show_message('Please choose first file')
        elif not self.data_frames[3]:
            self.show_message('Please choose second file')
        else:
            write_params_to_file()

            self.data_frames_for_analysis[0] = self.data_frames[0].between_time(self.data_frame_parameters[0][1],
                                                                                self.data_frame_parameters[0][2])
            self.data_frames_for_analysis[1] = self.data_frames[1].between_time(self.data_frame_parameters[1][1],
                                                                                self.data_frame_parameters[1][2])

            thread_animate = Thread(target=animate)
            thread_segment_1 = Thread(target=segment_data, args=([True]))
            thread_segment_2 = Thread(target=segment_data, args=([False]))
            thread_statistics = Thread(target=calculate_statistics)
            self.threads = [thread_segment_1, thread_segment_2, thread_animate, thread_statistics]

            thread_animate.start()
            thread_segment_1.start()
            thread_segment_2.start()

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
        elif file_name.__contains__('R'):
            content.append('Right Hand')

        if file_name.__contains__('1'):
            content.append('1 Week Data')
        elif file_name.__contains__('2'):
            content.append('2 Week Data')
        else:
            content.append('Data')

        title = "{} {} {} {}".format(content[0], content[1], content[2], content[3])

        return title

    def show_plot(self, is_first, data_frames):

        def create_plot(data_frame, figure, title):

            pyplt.figure(figure)
            pyplt.plot(data_frame['x'], color='#951732')
            pyplt.plot(data_frame['y'], color='#0b4545')
            pyplt.plot(data_frame['z'], color='#50AC3A')
            pyplt.xlabel('Time')
            pyplt.ylabel('Acceleration')
            pyplt.title(title)
            pyplt.legend(handles=[blue_patch, green_patch, red_patch], fontsize='x-small')
            pyplt.show()

        if is_first:
            if pyplt.fignum_exists(1):
                self.show_message('Figure 1 is already showing')
            else:
                title = self.create_title(self.file_names[0])
                create_plot(data_frames[0], 1, title)
        else:
            if pyplt.fignum_exists(2):
                self.show_message('Figure 2 is already showing')
            else:
                title = self.create_title(self.file_names[1])
                create_plot(data_frames[1], 2, title)

    def show_sets(self, file_name, is_first):

        def create_segment_plot(data_frame, figure, title, which_file):
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
                create_segment_plot(self.data_frames_for_analysis[which_file], figure, title, which_file=0)
        else:
            if pyplt.fignum_exists(4):
                self.show_message('Figure 4 is already showing')
            else:
                title = self.create_title(file_name)
                create_segment_plot(self.data_frames_for_analysis[which_file], figure, title, which_file=1)

    def show_statistics(self, is_first):

        if is_first:
            which_file = 0
        else:
            which_file = 1

        window = tk.Toplevel(self)

        outer_frame = ttk.Frame(window)
        outer_frame.pack(fill='both', expand=True, padx=15, pady=15)

        title = self.create_title(self.file_names[which_file])
        label_header = tk.Label(outer_frame, text=title)
        label_header.pack(side="top", fill="both", padx=10, pady=10)

        container = ttk.Frame(outer_frame)
        container.pack(fill='both', expand=True)

        statistics_tree = ttk.Treeview(container, columns=self.column_headers_for_statistics, show="headings")
        vsb = ttk.Scrollbar(container, orient="vertical", command=statistics_tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=statistics_tree.xview)
        statistics_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        statistics_tree.grid(column=0, row=0, sticky='nsew')
        vsb.grid(column=1, row=0, sticky='ns')
        hsb.grid(column=0, row=1, sticky='ew')

        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        self.build_tree(statistics_tree, self.data_frame_statistics, which_file)

    def compare_data(self):

        def compare(compare_by):

            data = self.data_frame_statistics
            training_set = data.sample(frac=0.8)
            test_set = data[~(data.isin(training_set)).all(1)]

            comp = ["", ""]
            if compare_by == "range":
                comp[0] = 'EType'
                comp[1] = 'Full Range'
            else:
                comp[0] = 'Hand'
                comp[1] = 'Left'

            label = []
            for index, row in training_set.iterrows():
                if row[comp[0]] == comp[1]:
                    label.insert(len(label), 0)
                else:
                    label.insert(len(label), 1)

            comp_data = {'training': training_set[training_set.columns[5:]].values,
                         'test': test_set[test_set.columns[5:]].values,
                         'label': label}
            comp_data = DotDict(comp_data)

            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(comp_data.training, comp_data.label)

            y_pred = clf.predict(comp_data.test)

            return training_set, test_set, y_pred, clf

        def summary_asper_compairing(compare_by, test, summary, classifier):

            if compare_by == "range":
                real_full = sum(test["EType"] == "Full Range")
                real_partial = sum(test["EType"] == "Partial Range")
                error_full = 0
                error_partial = 0

                for i in range(0, len(test)):
                    if test["EType"].values[i] == "Full Range":
                        if not test["EType"].values[i] == summary["Recognise As"].values[i]:
                            error_full += 1
                    else:
                        if not test["EType"].values[i] == summary["Recognise As"].values[i]:
                            error_partial += 1

                line1 = "{} {} {} {}".format(len(test), " out of ", len(self.data_frame_statistics),
                                             " has taken as samples\n")
                line2 = "{} {} {}".format("Number of full range samples    = ", real_full, "\n")
                line3 = "{} {} {}".format("Number of partial range samples = ", real_partial, "\n")
                line4 = "{} {} {} {}".format(error_full, " out of ", real_full,
                                             " Full Range exercise recognised as Partial Range\n")
                line5 = "{} {} {} {}".format(error_partial, " out of ", real_partial,
                                             " Partial Range exercise recognised as Full Range\n")
                line6 = "{} {} {}".format("Max depth of tree: ", classifier.tree_.max_depth, "\n")
            else:
                real_left = sum(test["Hand"] == "Left")
                real_right = sum(test["Hand"] == "Right")
                error_left = 0
                error_right = 0

                for i in range(0, len(test)):
                    if test["Hand"].values[i] == "Left":
                        if not test["Hand"].values[i] == summary["Recognise As"].values[i]:
                            error_left += 1
                    else:
                        if not test["Hand"].values[i] == summary["Recognise As"].values[i]:
                            error_right += 1

                line1 = "{} {} {} {}".format(len(test), " out of ", len(self.data_frame_statistics),
                                             " has taken as samples\n")
                line2 = "{} {} {}".format("Number of left hand samples    = ", real_left, "\n")
                line3 = "{} {} {}".format("Number of right samples = ", real_right, "\n")
                line4 = "{} {} {} {}".format(error_left, " out of ", real_left,
                                             " Left Hand recognised as Right Hand\n")
                line5 = "{} {} {} {}".format(error_right, " out of ", real_right,
                                             " Right Hand recognised as Left Hand\n")
                line6 = "{} {} {}".format("Max depth of tree: ", classifier.tree_.max_depth, "\n")

            lines = [line1, line2, line3, line4, line5, line6]

            return lines

        def show_training_set(data):

            window = tk.Toplevel(self)
            outer_frame = ttk.Frame(window)
            outer_frame.pack(fill='both', expand=True, padx=15, pady=15)

            label_for_training = tk.Label(outer_frame, text="Training Set")
            label_for_training.pack(side="top", fill="both", padx=10, pady=10)

            container_for_training = ttk.Frame(outer_frame, relief='groove')
            container_for_training.pack(fill='both', expand=True)

            tree_for_training = ttk.Treeview(container_for_training, columns=self.column_headers_for_statistics,
                                             show="headings")
            vsb_for_training = ttk.Scrollbar(container_for_training, orient="vertical", command=tree_for_training.yview)
            hsb_for_training = ttk.Scrollbar(container_for_training, orient="horizontal",
                                             command=tree_for_training.xview)
            tree_for_training.configure(yscrollcommand=vsb_for_training.set, xscrollcommand=hsb_for_training.set)
            tree_for_training.grid(column=0, row=0, sticky='nsew')
            vsb_for_training.grid(column=1, row=0, sticky='ns')
            hsb_for_training.grid(column=0, row=1, sticky='ew')

            container_for_training.grid_columnconfigure(0, weight=1)
            container_for_training.grid_rowconfigure(0, weight=1)

            text = "{} {}".format("Total number of samples =", len(data))

            label_training_summary = tk.Label(outer_frame, text=text)
            label_training_summary.pack(side="left", fill="both", padx=10, pady=10)

            self.build_tree(tree_for_training, data.sort_index(), 2)

        def show_test_set(data):

            window = tk.Toplevel(self)
            outer_frame = ttk.Frame(window)
            outer_frame.pack(fill='both', expand=True, padx=15, pady=15)

            label_for_test = tk.Label(outer_frame, text="Test Set")
            label_for_test.pack(side="top", fill="both", padx=10, pady=10)

            container_for_test = ttk.Frame(outer_frame, height=125, relief='groove')
            container_for_test.pack(fill='both', expand=True)

            tree_for_test = ttk.Treeview(container_for_test, columns=self.column_headers_for_statistics,
                                         show="headings")
            vsb_for_test = ttk.Scrollbar(container_for_test, orient="vertical", command=tree_for_test.yview)
            hsb_for_test = ttk.Scrollbar(container_for_test, orient="horizontal", command=tree_for_test.xview)
            tree_for_test.configure(yscrollcommand=vsb_for_test.set, xscrollcommand=hsb_for_test.set)
            tree_for_test.grid(column=0, row=0, sticky='nsew')
            vsb_for_test.grid(column=1, row=0, sticky='ns')
            hsb_for_test.grid(column=0, row=1, sticky='ew')

            container_for_test.grid_columnconfigure(0, weight=1)
            container_for_test.grid_rowconfigure(0, weight=1)

            text = "{} {}".format("Total number of samples =", len(data))

            label_test_summary = tk.Label(outer_frame, text=text)
            label_test_summary.pack(side="left", fill="both", padx=10, pady=10)

            self.build_tree(tree_for_test, data.sort_index(), 2)

        flag_f = 0
        for i in range(0, 2):
            if self.file_names[i][0] == "f":
                flag_f += 1
        flag_p = 0
        for i in range(0, 2):
            if self.file_names[i][0] == "p":
                flag_p += 1

        if flag_f == 2:
            compare_by = "hand"
        elif flag_p == 2:
            compare_by = "hand"
        else:
            compare_by = "range"

        training, test, y_pred, classifier = compare(compare_by)

        summary = DataFrame()
        summary["Set"] = test["Set"]
        summary["Repetition"] = test["Rep"]

        if compare_by == "range":
            summary["Exercise Type"] = test["EType"]
            column_for_summary = ["Set", "Repetition", "Exercise Type", "Recognise As"]
            result = []
            for i, pred in enumerate(y_pred):
                if pred == 0:
                    result.insert(len(result), "Full Range")
                else:
                    result.insert(len(result), "Partial Range")
        else:
            summary["Hand Type"] = test["Hand"]
            column_for_summary = ["Set", "Repetition", "Hand Type", "Recognise As"]
            result = []
            for i, pred in enumerate(y_pred):
                if pred == 0:
                    result.insert(len(result), "Left")
                else:
                    result.insert(len(result), "Right")

        summary["Recognise As"] = result

        window = tk.Toplevel(self)
        window.resizable(False, False)

        frame_top = ttk.Frame(window)
        frame_top.pack(fill="both", expand=True, padx=15, pady=15)

        label_summary_header = tk.Label(frame_top, text="Result of Decision Tree")
        label_summary_header.pack(side="top", fill="both", padx=10, pady=10)

        container_for_summary = ttk.Frame(frame_top, relief='groove')
        container_for_summary.pack(fill="both", expand=True)

        tree_for_summary = ttk.Treeview(container_for_summary, columns=column_for_summary, show="headings")
        vsb_for_summary = ttk.Scrollbar(container_for_summary, orient="vertical", command=tree_for_summary.yview)
        hsb_for_summary = ttk.Scrollbar(container_for_summary, orient="horizontal", command=tree_for_summary.xview)
        tree_for_summary.configure(yscrollcommand=vsb_for_summary.set, xscrollcommand=hsb_for_summary.set)
        tree_for_summary.grid(column=0, row=0, sticky='nsew')
        vsb_for_summary.grid(column=1, row=0, sticky='ns')
        hsb_for_summary.grid(column=0, row=1, sticky='ew')

        container_for_summary.grid_columnconfigure(0, weight=1)
        container_for_summary.grid_rowconfigure(0, weight=1)

        line = summary_asper_compairing(compare_by, test, summary, classifier)

        text = "{} {} {} {} {} {}".format(line[0], line[1], line[2], line[3], line[4], line[5])

        frame_label = ttk.Frame(window)
        frame_label.pack(fill='both', expand=True, pady=5, padx=10)

        label_header = tk.Label(frame_label, text="Summary", font=LARGE_FONT, justify="center")
        label_header.pack(fill='both', expand=True)

        label_summary_result = tk.Label(frame_label, text=text, justify='left')
        label_summary_result.pack(side="left", fill='both', expand=False)

        frame_button = ttk.Frame(window)
        frame_button.pack(expand=True, padx=10, pady=10)

        btn_show_training = ttk.Button(frame_button, text="Show Traning Set",
                                       command=lambda: show_training_set(training))
        btn_show_training.grid(column=0, row=0, sticky='ew')

        btn_show_test = ttk.Button(frame_button, text="Show Test Set",
                                   command=lambda: show_test_set(test))
        btn_show_test.grid(column=1, row=0, sticky='ew')

        self.build_tree(tree_for_summary, summary, 2)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StartPage(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.controller = controller
        self.parent = parent

        outer_frame = tk.Frame(self)
        outer_frame.pack(padx=50, pady=50)

        inner_frame = tk.Frame(outer_frame)
        inner_frame.grid_propagate()
        inner_frame.grid(row=0, column=0, columnspan=2, pady=10)

        label_greeting = tk.Label(inner_frame, text="Please Choose Files", font=LARGE_FONT)
        label_greeting.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)

        inner_left = tk.Frame(inner_frame)
        inner_left.grid(row=1, column=0, sticky='ew', padx=15)

        inner_right = tk.Frame(inner_frame)
        inner_right.grid(row=1, column=1, sticky='ew')

        btn_first_file = ttk.Button(inner_left, text="First File", compound=tk.LEFT,
                                    command=lambda: controller.get_file(True, label_first_file, label_first_param))
        btn_first_file.grid(row=1, column=0)

        label_first_file = ttk.Label(inner_left, image=controller.error)
        label_first_file.grid(row=1, column=1)

        btn_first_param = ttk.Button(inner_right, text="Parameters", compound=tk.LEFT,
                                     command=lambda: controller.parameters(True, label_first_param))
        btn_first_param.grid(row=1, column=2)

        label_first_param = ttk.Label(inner_right, image=controller.waiting)
        label_first_param.grid(row=1, column=3)

        btn_second_file = ttk.Button(inner_left, text="Second File ", compound=tk.LEFT,
                                     command=lambda: controller.get_file(False, label_second_file, label_second_param))
        btn_second_file.grid(row=2, column=0)

        label_second_file = ttk.Label(inner_left, image=controller.error)
        label_second_file.grid(row=2, column=1)

        btn_second_param = ttk.Button(inner_right, text="Parameters", compound=tk.LEFT,
                                      command=lambda: controller.parameters(False, label_second_param))
        btn_second_param.grid(row=2, column=2)

        label_second_param = ttk.Label(inner_right, image=controller.waiting)
        label_second_param.grid(row=2, column=3)

        label_waiting = ttk.Label(outer_frame)
        label_waiting.grid(row=3, column=0, columnspan=2)

        btn_start_analysis = ttk.Button(outer_frame, text="Start Analysis",
                                        command=lambda: controller.start_analysis(label_waiting))

        btn_start_analysis.grid(row=2, column=0, columnspan=2)


class AnalysisPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.parent = parent

        outer_frame = tk.Frame(self)
        outer_frame.pack(padx=50, pady=50)

        inner_frame = tk.Frame(outer_frame)
        inner_frame.grid(row=1, column=0, pady=10)

        top_frame = tk.Frame(inner_frame)
        top_frame.grid(row=0, column=0, padx=5)

        middle_frame = tk.Frame(inner_frame)
        middle_frame.grid(row=0, column=1, padx=5)

        right_frame = ttk.Frame(inner_frame)
        right_frame.grid(row=0, column=2, padx=5, pady=5)

        ttk.Separator(inner_frame).grid(row=1, column=0, sticky="ew", columnspan=3, pady=5)

        bottom_frame = ttk.Frame(inner_frame)
        bottom_frame.grid(row=2, column=0, columnspan=3)

        label_first = ttk.Label(top_frame, text="For First Dataset", font=LARGE_FONT)
        label_first.grid(row=0, column=0, columnspan=2, sticky='ew')

        btn_plot_data_first = ttk.Button(top_frame, text="Plot Data",
                                         command=lambda: controller.show_plot(True,
                                                                              controller.data_frames_for_analysis))
        btn_plot_data_first.grid(row=1, column=0, columnspan=2, sticky='ew')

        btn_show_sets_first = ttk.Button(top_frame, text="Show Sets",
                                         command=lambda: controller.show_sets(controller.file_names[0], True))
        btn_show_sets_first.grid(row=2, column=0, columnspan=2, sticky='ew')

        btn_statistics_first = ttk.Button(top_frame, text="Statistics",
                                          command=lambda: controller.show_statistics(True))
        btn_statistics_first.grid(row=3, column=0, columnspan=2, sticky='ew')

        label_empty = ttk.Label(middle_frame)
        label_empty.grid(row=0, column=0, pady=2)

        label_plot = ttk.Label(middle_frame, image=controller.graph)
        label_plot.grid(row=1, column=0, pady=2)

        label_segment = ttk.Label(middle_frame, image=controller.segment)
        label_segment.grid(row=2, column=0, pady=4)

        label_statistics = ttk.Label(middle_frame, image=controller.statistics)
        label_statistics.grid(row=3, column=0, pady=2)

        label_second = ttk.Label(right_frame, text="For Second Dataset", font=LARGE_FONT)
        label_second.grid(row=0, column=0, columnspan=2, sticky='ew')

        btn_plot_data_second = ttk.Button(right_frame, text="Plot Data",
                                          command=lambda: controller.show_plot(False,
                                                                               controller.data_frames_for_analysis))
        btn_plot_data_second.grid(row=1, column=0, columnspan=2, sticky='ew')

        btn_show_sets_second = ttk.Button(right_frame, text="Show Sets",
                                          command=lambda: controller.show_sets(controller.file_names[1], False))
        btn_show_sets_second.grid(row=2, column=0, columnspan=2, sticky='ew')

        btn_statistics_second = ttk.Button(right_frame, text="Statistics",
                                           command=lambda: controller.show_statistics(False))
        btn_statistics_second.grid(row=3, column=0, columnspan=2, sticky='ew')

        btn_compare = ttk.Button(bottom_frame, text="Compare", image=controller.compare, compound=tk.RIGHT,
                                 command=controller.compare_data)
        btn_compare.grid(row=0, column=0)

        btn_restart = ttk.Button(bottom_frame, image=controller.restart,
                                 command=controller.restart_test)
        btn_restart.grid(row=0, column=1)


app = ProjectApp()
app.resizable(False, False)
app.mainloop()

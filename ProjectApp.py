import datetime as dt
import os
import sys
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import time
from threading import Thread
import numpy as np

import matplotlib as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as pyplt
from matplotlib import style
from pandas import DataFrame
from scipy import signal

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
        self.data_frames = [None, None, False, False]
        self.data_frames_for_analysis = [None, None, False, False]
        self.data_frame_segments = [[], [], False, False]
        self.data_frame_statistics = [[], []]
        self.data_frame_transformed = [[], []]
        self.column_headers = ["Sets", "Mx", "My", "Mz", "STDx", "STDy", "STDz", "AMP1y", "AMP2y", "AMP3y"]
        self.program_path = os.getcwd()
        self.process_counter = 0
        self.allow_thread_3 = True

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

    def restart_test(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def show_filter(self, is_first):
        if is_first:
            which_file = 0
        else:
            which_file = 1

        b, a = signal.butter(4, 0.05, btype='lowpass')
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

        def segment_data(file_name, is_first, set_time):

            def get_periodics(data_frame, which_file, set_time, rest_time, limit, lag_time, is_filter):

                def get_cut_points(data, limit, set_time, rest_time):
                    cut_points = []
                    for i in range(0, 10):
                        if limit < 0:
                            start_index = np.where(data < limit)[0][0]
                        else:
                            start_index = np.where(data > limit)[0][0]
                        if i == 0:
                            cut_points.append(start_index)
                        else:
                            cut_points.append(cut_points[i - 1] + set_time * 100 + rest_time * 100 + start_index)
                        cut_point = start_index + set_time * 100 + rest_time * 100
                        data = data[cut_point::]

                    return cut_points

                if is_filter:
                    b, a = signal.butter(4, 0.05, btype='lowpass')
                    filtered_data = signal.filtfilt(b, a, data_frame)
                else:
                    filtered_data = np.array(data_frame)

                time_indexes = self.data_frames_for_analysis[which_file].index.tolist()
                start_time = time_indexes[0]
                data_frame = self.data_frames_for_analysis[which_file]
                cut_points = get_cut_points(filtered_data, limit, set_time, rest_time)
                lag = 0
                for i in range(0, 10):
                    periodic_start = start_time.to_datetime() + dt.timedelta(seconds=(cut_points[i] - 200) / 100 - lag)
                    periodic_end = periodic_start + dt.timedelta(seconds=set_time)
                    periodic = data_frame.between_time(periodic_start.time(), periodic_end.time())
                    self.data_frame_segments[which_file].append(periodic)
                    lag += lag_time

            if is_first:
                which_file = 0
            else:
                which_file = 1

            self.data_frame_segments[which_file + 2] = True

            if file_name.__contains__('bench'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.25, lag_time=3, is_filter=True)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=140,
                                      limit=-1.25, lag_time=2.5, is_filter=True)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.25, lag_time=0.5, is_filter=True)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=140,
                                      limit=-1.25, lag_time=0.5, is_filter=True)
            elif file_name.__contains__('military'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.20, lag_time=3, is_filter=True)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.20, lag_time=3, is_filter=True)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.22, lag_time=0, is_filter=True)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.22, lag_time=0.5, is_filter=True)
            elif file_name.__contains__('curl'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=150,
                                      limit=-0.5, lag_time=3, is_filter=True)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=150,
                                      limit=-0.1, lag_time=3, is_filter=True)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=150,
                                      limit=-0.45, lag_time=1, is_filter=True)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=150,
                                      limit=-0.1, lag_time=1, is_filter=True)
            elif file_name.__contains__('deadlift'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=200,
                                      limit=3.3, lag_time=4, is_filter=False)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=210,
                                      limit=3, lag_time=4, is_filter=False)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=210,
                                      limit=3.3, lag_time=0.5, is_filter=False)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=210,
                                      limit=3, lag_time=0.5, is_filter=False)
            elif file_name.__contains__('squat'):
                if file_name.__contains__('L'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.20, lag_time=4, is_filter=True)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=150,
                                      limit=-1.10, lag_time=3, is_filter=True)
                elif file_name.__contains__('R'):
                    if file_name[0] == 'f':
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.25, lag_time=1, is_filter=True)
                    else:
                        get_periodics(self.data_frames_for_analysis[which_file].y, which_file, set_time, rest_time=120,
                                      limit=-1.25, lag_time=1, is_filter=True)
            self.process_counter += 1

            if self.process_counter == 2:
                self.allow_thread_3 = False

        def calculate_statistics():

            def calculate_fft(which_file):

                peaks = []

                for i, segment in enumerate(self.data_frame_segments[which_file]):
                    transformed_data = np.fft.fft(segment.y)
                    transformed_real = transformed_data[:transformed_data.size / 2]
                    self.data_frame_transformed[which_file].append(transformed_real)
                    # TODO mph ve threshold a bak bakalım (, threshold=50)
                    segment_peaks = detect_peaks(transformed_real, mph=1)
                    integer_part = transformed_real[segment_peaks[:3]]
                    peaks.append(int(i) for i in integer_part)

                return peaks

            while self.allow_thread_3:
                "Waiting"
            self.process_counter += 1

            if self.process_counter == 3:
                time.sleep(3)
                for which_file in range(0, 2):
                    powers = calculate_fft(which_file)
                    for xi, set in enumerate(self.data_frame_segments[which_file]):
                        statistics = []
                        means = np.mean(set)
                        stds = np.std(set)

                        statistics.insert(0, xi + 1)
                        statistics.extend(means)
                        statistics.extend(stds)
                        statistics.extend(powers[xi])

                        self.data_frame_statistics[which_file].append(statistics)

                self.show_frame(AnalysisPage)

        def animate():
            gif_index = 0
            while True:
                try:
                    time.sleep(0.04)
                    img = tk.PhotoImage(file=self.program_path + "\loader.gif",
                                        format="gif - {}".format(gif_index))

                    label_waiting.config(text="Analysing Data")
                    label_waiting.config(image=img)
                    label_waiting.config(compound="right")
                    label_waiting.image = img

                    gif_index += 1
                except:
                    gif_index = 0

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

                thread_animate = Thread(target=animate)
                thread_segment_1 = Thread(target=segment_data, args=(self.file_names[0], True, set_time_1))
                thread_segment_2 = Thread(target=segment_data, args=(self.file_names[1], False, set_time_2))
                thread_statistics = Thread(target=calculate_statistics)

                thread_animate.start()
                thread_segment_1.start()
                thread_segment_2.start()
                thread_statistics.start()

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

    def show_transformed(self, is_first):

        if is_first:
            which_file = 0
        else:
            which_file = 1

        pyplt.figure(which_file + 1)
        gs = gridspec.GridSpec(5, 2)
        k = 0
        for i in range(0, 5):
            for j in range(0, 2):
                segment_plot = pyplt.subplot(gs[i, j])
                segment_plot.plot(self.data_frame_transformed[which_file][k])
                k += 1
        pyplt.show()

    def show_statistics(self, is_first):

        def _build_tree(tree, which_file):
            for col in self.column_headers:
                tree.heading(col, text=col.title())
                # adjust the column's width to the header string
                tree.column(col, width=tkFont.Font().measure(col.title()))
            for set in self.data_frame_statistics[which_file]:
                tree.insert('', 'end', values=set)
                # adjust column's width if necessary to fit each value
                for ix, val in enumerate(set):
                    col_w = tkFont.Font().measure(str(val))
                    if tree.column(self.column_headers[ix], width=None) < col_w:
                        tree.column(self.column_headers[ix], width=col_w - 50)

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

        statistics_tree = ttk.Treeview(container, columns=self.column_headers, show="headings")
        vsb = ttk.Scrollbar(container, orient="vertical", command=statistics_tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=statistics_tree.xview)
        statistics_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        statistics_tree.grid(column=0, row=0, sticky='nsew')
        vsb.grid(column=1, row=0, sticky='ns')
        hsb.grid(column=0, row=1, sticky='ew')

        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)

        _build_tree(statistics_tree, which_file)


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

        file = open(controller.program_path + r"\remember.txt", "r")
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

        btn_start_analysis = ttk.Button(outer_frame, text="Start Analysis",
                                        command=lambda: controller.start_analysis(label_waiting, entry_start_first,
                                                                                  entry_end_first, entry_start_second,
                                                                                  entry_end_second,
                                                                                  entry_set_time_first,
                                                                                  entry_set_time_second))
        btn_start_analysis.grid(row=2, column=0, columnspan=2)


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
        btn_show_sets_first.grid(row=2, column=0, columnspan=2, sticky='ew')

        btn_statistics_first = ttk.Button(left_frame, text="Statistics",
                                          command=lambda: controller.show_statistics(True))
        btn_statistics_first.grid(row=3, column=0, columnspan=2, sticky='ew')

        btn_show_transformed_first = ttk.Button(left_frame, text='Show Transformed',
                                                command=lambda: controller.show_transformed(True))
        btn_show_transformed_first.grid(row=4, column=0, columnspan=2, sticky='ew')

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

        btn_show_transformed_second = ttk.Button(right_frame, text='Show Transformed',
                                                 command=lambda: controller.show_transformed(False))
        btn_show_transformed_second.grid(row=4, column=0, columnspan=2, sticky='ew')

        btn_compare = ttk.Button(bottom_frame, text="Compare")
        btn_compare.grid(row=0, column=0)

        btn_restart = ttk.Button(bottom_frame, text="Restart Test", command=controller.restart_test)
        btn_restart.grid(row=0, column=1)


app = ProjectApp()
app.resizable(False, False)
app.mainloop()

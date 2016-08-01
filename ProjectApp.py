import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import matplotlib
matplotlib.use("TkAgg")
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

LARGE_FONT = ("Verdana", 12)


class ProjectApp(tk.Tk):  # The object inside the bracket is basically inherits the tkinter objects

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.filepath = ""

        tk.Tk.wm_title(self, "Accelerometer Analysis")

        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PlotPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

    def get_file(self):

        filepath = askopenfilename()

        if filepath:
            self.filepath = filepath


class StartPage(tk.Frame):

    def __init__(self, parent, controller, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.controller = controller
        self.parent = parent

        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        btn_getfile = ttk.Button(self, text="File Chooser", command=self.controller.get_file)
        btn_getfile.pack()

        btn_plotpage = ttk.Button(self, text="Show Plot", command=lambda: controller.show_frame(PlotPage))
        btn_plotpage.pack()


class PlotPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.parent = parent

        label = tk.Label(self, text="Ploting page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        btn_back = ttk.Button(self, text="Back Home", command=lambda: controller.show_frame(StartPage))
        btn_back.pack()

        btn_show = ttk.Button(self, text="Show Plot", command=self.show_plot)
        btn_show.pack()

    def show_plot(self):
        
        data_frame = DataFrame.from_csv(self.controller.filepath)

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(data_frame)

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    '''
    try:
        with askopenfilename() as filename:   # show an "Open" dialog box and return the path to the selected file
            data_frame = DataFrame.from_csv(filename)
    except:
        print("Wrong file type")
    '''

app = ProjectApp()
app.geometry("500x500")
app.mainloop()

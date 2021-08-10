import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
# from tkcalendar import DateEntry
from datetime import datetime
from dog_face_detector import DogFaceDetector
from dog_predictor import predict_dog

# dfd = DogFaceDetector()

LARGE_FONT= ("Verdana", 12)
plt.style.use("ggplot")


### Tkinter Setup:
root = tk.Tk
root_frame = tk.Frame

class DYapp(root):
    def __init__(self, *args, **kwargs):
        root.__init__(self, *args, **kwargs)
        root.wm_title(self, "DogYears Dog Age Predictor")
        container = root_frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        ### For multiple pages
        # for F in (StartPage, BothPlots):#, TimeStampPlots, SamplePlots):
        #     frame = F(container, self)
        #     self.frames[F] = frame
        #     frame.grid(row=0, column=0, sticky="NESW")
        ### For just start page
        frame = StartPage(container, self)
        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky="NESW")
        self.show_frame(StartPage)
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()  # raises to front

class StartPage(root_frame):
    def __init__(self, parent, controller):
        root_frame.__init__(self, parent)
        self.parent = parent
        self.page_title = tk.StringVar()
        self.page_title.set('SELECT A FILE')
        page_title_label = ttk.Label(self, textvariable=self.page_title,
                                     font="LARGE_FONT").grid(row=0, column=0, columnspan=4)  # .pack()
        self.select_file()
    def select_file(self):
        self.filepath = tk.StringVar()
        self.filepath.set('Select a File')
        browse_button = ttk.Button(self, textvariable=self.filepath,
                                   command=self.show_file).grid(row=1, column=0, columnspan=4)  # .pack()
    def browse_file(self): ### Fix
        self.filename = askopenfilename(
                                        # filetypes=(
                                                    # ("JPG (*.jpg)", "*.jpg*"), 
                                                    # ("JPEG (*.jpeg)", "*.jpeg*"), 
                                                    # ("PNG (*.png)", "*.png*"), 
                                                    # ("All Files (*.*)", "*.*"))
                                                    )
        return self.filename
    def show_file(self):
        file = self.browse_file()
        if file != '':
            self.filepath.set(file.split('/')[-1])
            self.prediction = predict_dog(file)
            self.page_title.set(self.prediction)
            self.show_img()
    def show_img(self):
        img = cv2.cvtColor(cv2.imread(self.filename), cv2.COLOR_BGR2RGB)
        # fig = plt.figure(1)
        # plt.ion()
        # plt.imshow(img)

        self.f = Figure(figsize=(5,5), dpi=100)
        self.a = self.f.add_subplot(111)
        self.a.imshow(img)
        self.a.axis('off')

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, column=0)
        # canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # plt.show()


# def dogface_predictor(img_path):
#     df_img = dfd.get_dogface(img_path, predict_features=True, save=False)[0]

#     plt.imshow(dfd.img_result)
#     plt.show()
# dogface_predictor(img_path)
### Launching the app
app = DYapp()
if __name__ == '__main__':
    app.mainloop()
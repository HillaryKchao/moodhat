import queue
import threading
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pythonosc import dispatcher, osc_server
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path


# Determine the directory containing the module
relative_path = Path("../BCI.py")
module_directory = relative_path.parent.resolve()

# Add the directory to sys.path
sys.path.insert(0, str(module_directory))

# Now you can import the module
from BCI import *

firsttime = None # record the first timestamp during recording


headset = BCI("MuseS") # ["TP9", "AF7", "AF8", "TP10"]
headset_server, headset_server_thread = headset.launch_server()
headset.load_lsl_data("../EEG_recording_2024-11-16-22.34.56.csv")

pipe_obj = Pipe(2, headset.no_of_channels, headset.store)
pipe_obj.launch_server()

# creating processing blocks
# notch_filter = NotchFilter(headset.no_of_channels, headset.store, 30, headset.sampling_rate, 1)
# low_pass_filter = LowPassFilter(headset.no_of_channels, headset.store, 60, headset.sampling_rate)
high_pass_filter = HighPassFilter(headset.no_of_channels, headset.store, 30, headset.sampling_rate)

# pipe_obj.store = notch_filter.action()
# pipe_obj.store = low_pass_filter.action()
pipe_obj.store = high_pass_filter.action()

# Real-time plotting setup

def store_empty(store_list):
    return any([item.empty() for item in store_list])

final_output_store = pipe_obj.store
number_of_plots = pipe_obj.no_of_outputs
number_of_lines = pipe_obj.no_of_input_channels
list_of_labels = headset.channel_names

axes_list = [None] * number_of_plots
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
axes_list = [ax1, ax2]


xdata = [[] for _ in range(number_of_plots)]
ydata_matrix = [[[] for _ in range(number_of_lines)] for _ in range(number_of_plots)]
plot_matrix = [[None for _ in range(number_of_lines)] for _ in range(number_of_plots)]


for i in range(number_of_plots):
    for j in range(number_of_lines):
        a, = axes_list[i].plot([],[], label=list_of_labels[j])
        plot_matrix[i][j] = a
    axes_list[i].legend()

def init():
    for i in range(number_of_plots):
        axes_list[i].set_xlim(0, 40)
        axes_list[i].set_ylim(-1200, 1200)
        for j in range(number_of_lines):
            plot_matrix[i][j].set_data([], [])
    return [line for sublist in plot_matrix for line in sublist]

def update(frame):
    global firsttime
    for i in range(number_of_plots):
        if not store_empty(final_output_store[i]):
            
            for j in range(number_of_lines):
                ydata_matrix[i][j].append(final_output_store[i][j].get())
            xdata[i].append(frame)

            if len(xdata[i]) > 40:  # Change the window size if needed
                xdata[i].pop(0)
                for j in range(number_of_lines):
                    ydata_matrix[i][j].pop(0)

            axes_list[i].set_xlim(xdata[i][0], xdata[i][-1])

            for j in range(number_of_lines):
                plot_matrix[i][j].set_data(xdata[i], ydata_matrix[i][j])
        

ani = animation.FuncAnimation(fig, update, frames=range(5000), init_func=init, interval=1)
plt.show()

# saving animation to gif
writer = animation.PillowWriter(fps=60, metadata=dict(artist='Me'), bitrate=100)
ani.save('scatter.gif', writer=writer)

try:
    while True:
        pass  # Keep the main thread running
except KeyboardInterrupt:
    print("Exiting...")
    headset_server.shutdown()
    headset_server_thread.join()
    plt.close()
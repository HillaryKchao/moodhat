import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue

from BCI import BCI, Pipe

bci = BCI()
bci.load_lsl_data("EEG_recording_2024-11-16-22.34.56.csv")      # data from ria sitting (neutral)
input_data = bci.store[3]       # TP10 channel

pipe = Pipe(1, 1, input_data)
pipe.action()
output_data = pipe.store

#making the graph!!
fig, axes = plt.subplots(2)
fig.suptitle("Pipe Class Animation: Input vs. Output Data")

x_axis_data = bci.time_store
input_data_list, output_data_list, x_axis_data_list = [], [], []

while not input_data.empty():
    input_data_list.append(input_data.get())
    output_data_list.append(output_data.get())
    x_axis_data_list.append(x_axis_data.get())

#Graph 1: input 
axes[0].set_title("Input Data")
axes[0].legend()

axes[0].plot(x_axis_data_list, input_data_list, 'b-', label='Input Data') 

#Graph 2: output 
axes[1].set_title("Output Data")
axes[1].legend()

axes[1].plot(x_axis_data, output_data_list, 'r-', label='Output Data')

#Creating Animation
# wtf is this
# ani = FuncAnimation(fig, update, frames=len(input_data), init_func=init, blit=True)
#i don't know what the updating funcanimation is

plt.show()
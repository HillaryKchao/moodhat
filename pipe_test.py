import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from BCI import Pipe 

x_axis_data = np.linspace(0, 20, 100)
input_data = np.sin(x) + np.random.normal(scale=0.1, size=100) #random generated sample data

pipe = Pipe()
output_data = pipe.process(input_data) #do we have a function process()????

#making the graph!!
fig, axes = plt.subplots(2)
fig.suptitle("Pipe Class Animation: Input vs. Output Data")

#Graph 1: input 
axes[0].set_title("Input Data")
axes[0].set_xlim(0, len(input_data))
axes[0].set_ylim(input_data.min() - 0.2, input_data.max() + 0.2)
axes[0].legend()

axes[0].plot(x_axis_data, input_data, 'b-', label='Input Data') 

#Graph 2: output 
axes[1].set_title("Output Data")
axes[1].set_xlim(0, len(output_data))
axes[1].set_ylim(output_data.min() - 0.2, output_data.max() + 0.2)
axes[1].legend()

axes[1].plot(x_axis_data, output_data, 'r-', label='Output Data')

#Creating Animation
# wtf is this
# ani = FuncAnimation(fig, update, frames=len(input_data), init_func=init, blit=True)
#i don't know what the updating funcanimation is

plt.show()
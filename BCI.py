import numpy as np
from abc import ABC, abstractmethod
import queue
import threading
# from utils import generate_random_string, nanVal
from scipy.integrate import simps
import scipy.signal as signal
import argparse
from pythonosc import dispatcher as disp, osc_server
import matplotlib.pyplot as plt
# import pylsl
# import mne

class BCI:
	'''
		Class to manage collection of the BCI data and store into a buffer store. Used as the input of
		the pipeline To receive data from the headset and store it into a list of storage queues for which
		later blocks of the pipeline can pull from.. 
		NOTE: This version is only set up for Muse S, using Petal Metrics' tool for handling the streaming protocol.
		The current version works with OSC, LSL streaming is a TODO.

		Class Variable:
			- sampling_rate: sampling rate of the device
			- streaming_software: software used for heandling the streaming (Petals for Muse S) (TODO: make this an option arguments)
			- streaming_protocol: streaming protocol used by headset
			- channel_names: list of names of channels, used for labels
			- no_of_channels: number of channels for the device. Channels are defined as an independent stream of data that most processing blocks will treat as independent streams to compute the same operations on in parallel
			- store: a list of queues where input data from the datastreams are added to, which later blocks can get() from
			- time_store: single queue that holds timestamp data # TODO: make this function
			- launch_server: function to initialize and start streaming data into self.store, determined by self.streaming_protocol

	'''
	def __init__(self, BCI_name="MuseS", BCI_params={}):
		self.name = BCI_name
		if self.name == "MuseS":
			self.BCI_params = {"sampling_rate": 256, "channel_names":["TP9", "AF7", "AF8", "TP10"], "streaming_software":"Petals", "streaming_protocol":"OSC", "cache_size":256*30}
			if BCI_params:
				for i,j in BCI_params:
					self.BCI_params[i] = j
		else:
			raise Exception("Unsupported BCI board") # change this when adding other headsets
		
		self.sampling_rate = self.BCI_params["sampling_rate"]
		self.streaming_software = self.BCI_params["streaming_software"] # # TODO: add as optional argument
		self.streaming_protocol = self.BCI_params["streaming_protocol"] # TODO: mandatory argument, add error handling
		self.channel_names = self.BCI_params["channel_names"] # TODO: add error handling -> warning for non standard channel names
		self.no_of_channels = len(self.channel_names) # TODO: add error handling
		self.store = [queue.Queue() for i in range(self.no_of_channels)]
		self.time_store = queue.Queue()
		if self.streaming_protocol == 'LSL':
			self.launch_server = self.launch_server_lsl
		elif self.streaming_protocol == 'OSC':
			self.launch_server = self.launch_server_osc
		

	def action(self):
		pass # not required for BCI object

	def handle_osc_message(self, address, *args):
		'''
		Receives messages through the OSC channel and adds them to a list of queues.
		''' 
		self.store[0].put(args[5])
		self.store[1].put(args[6])
		self.store[2].put(args[7])
		self.store[3].put(args[8])
		self.time_store.put(args[3] + args[4])

	def launch_server_osc(self):
		parser = argparse.ArgumentParser()
		parser.add_argument('-i', '--ip', type=str, required=False,
							default="127.0.0.1", help="The ip to listen on")
		parser.add_argument('-p', '--udp_port', type=str, required=False, default=14739,
							help="The UDP port to listen on")
		parser.add_argument('-t', '--topic', type=str, required=False,
							default='/PetalStream/eeg', help="The topic to print")
		args = parser.parse_args()

		dispatcher = disp.Dispatcher()
		dispatcher.map(args.topic, self.handle_osc_message)

		server = osc_server.ThreadingOSCUDPServer((args.ip, args.udp_port), dispatcher)
		server_thread = threading.Thread(target=server.serve_forever)
		server_thread.daemon = True
		server_thread.start()

		return server, server_thread
	
	def launch_server_lsl(self):
		'''
		Receives messages through the OSC channel and adds them to a list of queues.
		TODO: broken, requires fixing.
		''' 
		# TODO: take code from Petal metrics and implement streaming with LSL
		# parser = argparse.ArgumentParser()
		# parser.add_argument('-n', '--stream_name', type=str, required=True,
		# 					default='PetalStream_eeg', help='the name of the LSL stream')
		# args = parser.parse_args()
		# print(f'looking for a stream with name {args.stream_name}...')
		# streams = pylsl.resolve_stream('name', args.stream_name)
		# if len(streams) == 0:
		# 	raise RuntimeError(f'Found no LSL streams with name {args.stream_name}')
		# inlet = pylsl.StreamInlet(streams[0])

		...

	def load_lsl_data(self, csv_path):
		'''
		Loads data from the Excel file of data (collected by Muse using LSL streaming) into a list of queues.
		'''
		with open(csv_path, 'r') as csvfile:
			data_array = np.loadtxt(csvfile, delimiter=',', skiprows=1, usecols=(0, 1, 2, 3, 4))
		first_time = data_array[0][0]
		for channel_data in data_array:
			self.time_store.put(channel_data[0] - first_time)
			for i in range(self.no_of_channels):
				self.store[i].put(channel_data[i+1])

class Pipe:
	'''
	Connects one or more ProcessingBlocks together. Requires a knowledge of which pipes are connected together, the number of channels of data passed by each channel, and the number of targets the data has to go to. Pipes can be connected to multiple piples to duplicate data being sent across (???)

	Class variables:
		- no_of_outputs: number of outputs data from each channel goes to, i.e. how many outgoing connections does the same value have to go to
		- no_of_input_channels: number of channels (usually BCI channels) from the input
		- input_store: the store ( of type Queue / type with .get() ) from the input
		- name: name of the pipe (for ID and pipeline visualization)
		- store: list of "output" queues which the successive block in the pipeline will get data from
	'''
	def __init__(self, no_of_outputs, no_of_input_channels, input_store) -> None:
		self.no_of_outputs = no_of_outputs
		self.no_of_input_channels = no_of_input_channels # if the prior stage has multiple pipes, it should be indexed before being passed into the arguments of the constructor
		self.input_store = input_store
		self.name = "PIPE_" # + generate_random_string()
		# self.store -> stores the outputs for the pipe, maintains that the outputs of the pipe are used at the same time
		# should be indexed manually (can be done by a synthesizer to automate this) to access the correct path in the pipe
		self.store = [[queue.Queue() for j in range(no_of_input_channels)] for i in range(no_of_outputs)]
		

	def action(self):
		try:
			# loops through channels, gets the value for the corresponding channel and adds it to every self.store index
			# i -> output store number; j -> input channel number
		#TODO: run self.input_store through notch, low-pass, and high-pass (determine order) before storing each value
			#sample for running through high pass filter
			# HP_processsed = HighPassFilter(4, self.input_store, ....)
			# #same for low pass, notch MAKE SURE TO CHECK ORDER
			# LP_processsed = LowPassFilter(4, HP_processsed.store, ....)
			# Notch_processed = NotchFilter(4, LP_processsed.store, ....) 
			# while Notch_processed is not queue.Empty():
			# 	for j in range(Notch_processed.no_of_input_channels):
			# 		value = Notch_processed.input_store.get()
			# 		for i in range(self.no_of_outputs):
			# 			self.store[i][j].put(value)
			
			while self.input_store is not queue.Empty():
				for j in range(self.no_of_input_channels):
					value = self.input_store.get()
					for i in range(self.no_of_outputs):
						self.store[i][j].put(value)
		
		except KeyboardInterrupt:
			print(f"Closing {self.name} thread...")


	def launch_server(self):
		average_thread = threading.Thread(target=self.action)
		average_thread.daemon = True
		average_thread.start()
		print(f"{self.name} thread started...")

	# TODO: add functionality to stop the pipe if any output is not ready for loading yet

class csvOutput:
#ToDO: Implement this
	'''
	Takes incoming data and saves the data into a CSV format. 
	'''
	def __init__(self):
		...


class ProcessingBlock(ABC):
	'''
	Abstract class for processing block. Should have an _init_ function, an action function, a process function to start the processing thread
	'''
	# def __init__(self, cache_dim=(256*5)):
		# self.input_queue = input_queue # reference to queue to pool data from into own cache
		# self.output_queue = output_queue 

	@abstractmethod
	def __init__():
		pass

	def launch_server(self):
		average_thread = threading.Thread(target=self.action)
		average_thread.daemon = True
		average_thread.start()
		print(f"{self.name} thread started...")
	
	def action(self):
		pass




class MovingAverageFilter(ProcessingBlock):
	"""
	Calculates the moving average for a stream of data. Can handle multiple channels simultaneously. 

	Class variables:
		- window_size: how many input samples are averaged per output averaged sample
		- no_of_input_channels: number of channels (usually BCI channels) from the input
		- input_store: the store ( of type Queue / type with .get() ) from the input
		- name: name of the pipe (for ID and pipeline visualization)
		- store: list of "output" queues which the successive block in the pipeline will get data from
		- cache: place to hold values while processing them
		- valuestop: FIXME (not used yet)
		- index: current index in the cache to be filled
	"""
	# default -> stride = window_size
	# TODO: implement stride, window type, edge handling
	def __init__(self, no_of_input_channels, input_store, window_size=8):
		...

	def action(self):
		...




class PSD(ProcessingBlock): 
	"""
	Computes the PSD of incoming data stream, and output the PSD per channel.
	
	Class variables:
		- no_of_input_channels: number of channels (usually BCI channels) from the input
		- sampling_frequency: sampling frequency of the data
		- window_size: how many samples are used for the PSD
		- input_store: the store ( of type Queue / type with .get() ) from the input
		- name: name of the pipe (for ID and pipeline visualization)
		- frequency_resolution: gap in frequency of outputted PSD array 
		- no_of_unique_values: no. of unique valeus in the PSD array ouput (FIXME not used yet)
		- filled: is the cache filled yet (required for shift register system) (FIXME not used yet)
		- valid: identifies to the next block that the PSD outputs are valid or not (FIXME not used yet)
		- index: index for which part of the 
		- cache: place to hold values while processing them (shift register system)
		- store: list of "output" queues which the successive block in the pipeline will get data from
		
	"""
	def __init__(self, no_of_input_channels, sampling_frequency, window_size, input_store):
		assert len(input_store) == no_of_input_channels, "The input store must have one queue per channel."

		self.no_of_input_channels = no_of_input_channels
		self.sampling_frequency = sampling_frequency
		self.window_size = window_size
		self.input_store = input_store

	
	def compute_psd(self):
		"""
        Computes the Power Spectral Density (PSD) for each channel in the input store.
        
        :return: A list of PSDs for each channel. Each element is a tuple (frequencies, psd_values).
        """
		psd_results, channel_data = [], []

		for i in range(self.no_of_input_channels):
			while not self.input_store[i].empty():
				channel_data.append(self.input_store[i].get())
			if len(channel_data) < self.window_size:
				print(f"Warning: Channel {i} has fewer data points than the window size. Skipping PSD calculation.")
			
			# apply windowing function (Hanning window by default)
			print(self.window_size)
			window = np.hanning(self.window_size)
			signal_segment = np.array(channel_data[:self.window_size]) * window

			# computer the FFT of the signal segment
			fft_result = np.fft.fft(signal_segment)
			fft_freq = np.fft.fftfreq(self.window_size, 1/self.sampling_frequency)  # frequency bins
			
			# compute PSD (magnitude squared of the FFT)
			psd_values = np.abs(fft_result) ** 2
			psd_values = psd_values[:self.window_size // 2]  # keep only positive frequencies

			# only keep frequencies up to Nyquist frequency
			freqs = fft_freq[:self.window_size // 2]

			psd_results.append((freqs, psd_values))

		return psd_results
		
	
	def plot_psd(self):
		"""
        Plots the PSD for each channel.
        """
		psd_results = self.compute_psd()

		plt.figure(figsize=(10, 6))
        
		for i, (freqs, psd_values) in enumerate(psd_results):
			plt.semilogy(freqs, psd_values, label=f'Channel {i + 1}')

		plt.title('Power Spectral Density for Each Channel')
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Power Spectral Density (dB/Hz)')
		plt.legend()
		plt.grid(True)
		plt.show()

		


class ArrayIntegrator(ProcessingBlock):
	'''
	Integrates the array for each channel and outputs a single value per channel. Does not integrate over time

	Class Variables:
	- no_of_input_channels: number of input channels
	- x_axis_resolution: dx of the x-axis values, assumed uniform spacing
	- no_of_inputs: number of input values (without regarding channel count)
	- prior_valid: points to the validity of the prior block (need to be precomputed for validity of multiple inputs) (TODO: not implemented)
	- array_of_inputs <-> *args : list of stores to input from to integrate
	- name: name of block
	- cache: temporary store (no_of_input_channels x no_of_inputs)
	- store: output store (1 x no_of_input_channels)
	- valid: validity of output (TODO: not implemented)
	'''
	def __init__(self, no_of_input_channels, x_axis_resolution, no_of_inputs, prior_valid, *args):
		...



class Ratio(ProcessingBlock):
	'''
	Calculates the ratio between two values over multiple channels. Outputs input1 / input2

	Class variables:
	- no_of_input_channels: number of channels in inputs (must be equal)
	- input1: source for the numerator input
	- input2: source for the denominator input
	- store: queue for the output stream
	- valid: indicator for valid input
	'''
	def __init__(self, no_of_input_channels, input_store1, input_store2, prior_valid1, prior_valid2):
		...



class NotchFilter(ProcessingBlock):
	#TODO: copy and paste notchFilter implementation into this file or import it
	'''
	Applies a Notch Filter on the data stream. Apply this on the input data (easiest to do this before applying any other processing blocks)

	Class variables:
	- no_of_input_channels: number of channels in inputs (must be equal)
	'''
	def __init__(self, no_of_input_channels, input_store, notch_frequency, sampling_freqeuncy, quality_factor):
		...



#TODO: copy and paste high pass, low pass filter implementation into this file (same format as NotchFilter) or import them



	
	

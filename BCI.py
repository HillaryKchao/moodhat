import numpy as np
from abc import ABC, abstractmethod
import queue
import threading
from scipy.integrate import simps
import scipy.signal as signal
import argparse
from pythonosc import dispatcher as disp, osc_server
import matplotlib.pyplot as plt

from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions

# Handy little enum to make code more readable
class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

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
					value = self.input_store[j].get()
					for i in range(self.no_of_outputs):
						self.store[i][j].put(value)
		
		except KeyboardInterrupt:
			print(f"Closing {self.name} thread...")


	def launch_server(self):
		average_thread = threading.Thread(target=self.action)
		average_thread.daemon = True
		average_thread.start()
		print(f"{self.name} thread started...")

class ProcessingBlock(ABC):
	'''
	Abstract class for processing block. Should have an _init_ function, an action function, a process function to start the processing thread
	'''

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

class PSD(ProcessingBlock): 
	"""
	Computes the PSD of incoming data stream, and output the PSD per channel.
	
	Class variables:
		- no_of_input_channels: number of channels (usually BCI channels) from the input
		- sampling_frequency: sampling frequency of the data
		- window_size: how many samples are used for the PSD
		- input_store: the store ( of type Queue / type with .get() ) from the input
		
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
		
		Returns a list of PSDs for each channel. Each element is a tuple (frequencies, psd_values).
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

class NotchFilter(ProcessingBlock):
	#TODO: copy and paste notchFilter implementation into this file or import it
	'''
	Applies a Notch Filter on the data stream. Apply this on the input data (easiest to do this before applying any other processing blocks)

	Class variables:
	- no_of_input_channels: number of channels in inputs (must be equal)
	'''
	def __init__(self, no_of_input_channels, input_store, notch_frequency, sampling_frequency, bandwidth):
		self.no_of_input_channels = no_of_input_channels
		self.input_store = input_store
		self.notch_frequency = notch_frequency
		self.sampling_frequency = sampling_frequency
		self.bandwidth = bandwidth

		# -*- coding: utf-8 -*-


	def action(self):
		print('Looking for an EEG stream...')
		data = pd.read_csv('EEG_recording1.csv')  #
		if len(streams) == 0:
			raise RuntimeError('Can\'t find EEG stream.')

		# Set active EEG stream to inlet and apply time correction
		print("Start acquiring data")
		inlet = StreamInlet(streams[0], max_chunklen=12)
		eeg_time_correction = inlet.time_correction()

		# Get the stream info and description
		info = inlet.info()
		description = info.desc()

		# Get the sampling frequency
		fs = int(info.nominal_srate())

		# Initialize raw EEG data buffer
		eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
		filter_state = None  # for use with the notch filter

		# Compute the number of epochs in "buffer_length"
		n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
								SHIFT_LENGTH + 1))

		# Initialize the band power buffer (for plotting)
		# bands will be ordered: [delta, theta, alpha, beta]
		band_buffer = np.zeros((n_win_test, 4))

   		print('Press Ctrl-C in the console to break the while loop.')

		try:
			# The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
			while True:
				""" 3.1 ACQUIRE DATA """
				# Obtain EEG data from the LSL stream
				eeg_data, timestamp = inlet.pull_chunk(
					timeout=1, max_samples=int(SHIFT_LENGTH * fs))

				# Only keep the channel we're interested in
				ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

				# Update EEG buffer with the new data
				eeg_buffer, filter_state = utils.update_buffer(
					eeg_buffer, ch_data, notch=True,
					filter_state=filter_state)

				""" 3.2 COMPUTE BAND POWERS """
				# Get newest samples from the buffer
				data_epoch = utils.get_last_data(eeg_buffer, EPOCH_LENGTH * fs)

				# Compute band powers
				band_powers = utils.compute_band_powers(data_epoch, fs)
				band_buffer, _ = utils.update_buffer(band_buffer, np.asarray([band_powers]))
				# Compute the average band powers for all epochs in buffer
				# This helps to smooth out noise
				smooth_band_powers = np.mean(band_buffer, axis=0)
				
				""" 3.3 COMPUTE NEUROFEEDBACK METRICS """
				# These metrics could also be used to drive brain-computer interfaces

				# Alpha Protocol:
				# Simple redout of alpha power, divided by delta waves in order to rule out noise
				alpha_metric = smooth_band_powers[Band.Alpha] / \
					smooth_band_powers[Band.Delta]
				print('Alpha Relaxation: ', alpha_metric)

		except KeyboardInterrupt:
			print('Closing!')

	# def action(self):
	# 	nyquist = 0.5 * self.sampling_frequency
	# 	low = (self.notch_frequency - self.bandwidth / 2) / nyquist
	# 	high = (self.notch_frequency + self.bandwidth / 2) / nyquist
		
	# 	# Create a notch filter using iirfilter
	# 	b, a = signal.iirfilter(N=4, Wn=[low, high], btype='bandstop', ftype='butter')

	# 	# Filtrate the signal (zero-phase filter with filtfilt)
	# 	filtered_data = signal.filtfilt(b, a, self.input_store, axis=0)
	# 	return filtered_data

class LowPassFilter(ProcessingBlock):
	'''
	Apples a low pass filter on the data stream. 

	Class variables:
		- no_of_input_channels: number of channels (usually BCI channels) from the input
		- input_store: the store ( of type Queue / type with .get() ) from the input
		- low_pass_frequency: the chosen frequency in which all signals above it are attenuated
		- sampling_frequency: sampling frequency of the data
	'''

	def __init__(self, no_of_input_channels, input_store, low_pass_frequency, sampling_frequency):
		self.no_of_input_channels = no_of_input_channels
		self.input_store = input_store
		self.low_pass_frequency = low_pass_frequency
		self.sampling_frequency = sampling_frequency

	def action(self, order=5):
		nyquist = 0.5 * self.sampling_frequency
		normal_cutoff = self.low_pass_frequency / nyquist
		b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
		filtered_data = signal.filtfilt(b, a, self.input_store, padlen=len(self.input_store)-1)
		return filtered_data

class HighPassFilter:
	'''
	Apples a high pass filter on the data stream. 

	Class variables:
		- no_of_input_channels: number of channels (usually BCI channels) from the input
		- input_store: the store ( of type Queue / type with .get() ) from the input
		- high_pass_frequency: the chosen frequency in which all signals below it are attenuated
		- sampling_frequency: sampling frequency of the data
	'''
	def __init__(self, no_of_input_channels, input_store, high_frequency, sampling_frequency):
		self.no_of_input_channels = no_of_input_channels
		self.input_store = input_store
		self.high_frequency = high_frequency
		self.sampling_frequency = sampling_frequency
		
		self.store = [queue.Queue() for _ in range(no_of_input_channels)]
		self.b, self.a = signal.butter(4, high_frequency / (0.5 * sampling_frequency), btype='high', analog=False)

	def process(self, data):
		return signal.lfilter(self.b, self.a, data)

	def action(self):
		for ch in range(self.no_of_input_channels):
			if not self.input_store[ch].empty():
				sample = self.input_store[ch].get()
				filtered_sample = self.process([sample])[0]
				self.input_store[ch].put(filtered_sample)
		return self.input_store
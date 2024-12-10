import sys
from pathlib import Path

relative_path = Path("../BCI.py")
module_directory = relative_path.parent.resolve()

sys.path.insert(0, str(module_directory))

from BCI import *

headset = BCI("MuseS") # ["TP9", "AF7", "AF8", "TP10"]
headset_server, headset_server_thread = headset.launch_server()

headset.load_lsl_data("../EEG_recording_2024-11-16-22.34.56.csv")

# Generate a test signal: combination of low-frequency (0.5 Hz) and high-frequency (10 Hz) sine waves
def generate_test_signal(duration=5, sampling_rate=256):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    low_freq = np.sin(2 * np.pi * 0.5 * t)  # Low-frequency component (0.5 Hz)
    high_freq = np.sin(2 * np.pi * 10 * t)  # High-frequency component (10 Hz)
    signal = low_freq + high_freq
    return t, signal

def plot_high_pass(time, original, filtered):
		plt.figure(figsize=(12, 6))
		plt.plot(time, original, label="Original Signal (Low + High Frequency)", alpha=0.7)
		plt.plot(time, filtered, label="Filtered Signal (High-Pass Applied)", linewidth=2)
		plt.xlabel("Time (s)")
		plt.ylabel("Amplitude")
		plt.title("High-Pass Filter Test")
		plt.legend()
		plt.grid(True)
		plt.show()

def test_high_pass_filter():
    # Test parameters
    sampling_rate = 256  # Hz
    duration = 5  # seconds
    cutoff_frequency = 1  # Hz (high-pass filter cutoff)

    # Generate test signal
    time, test_signal = generate_test_signal(duration, sampling_rate)

    # Set up input and output queues
    input_store = [queue.Queue() for _ in range(1)]  # Single-channel test

    # Push test signal into input store
    for sample in test_signal:
        input_store[0].put(sample)

    # Initialize and run HighPassFilter
    high_pass_filter = HighPassFilter(
        no_of_input_channels=1,
        input_store=input_store,
        cutoff_frequency=cutoff_frequency,
        sampling_frequency=sampling_rate,
    )

    # Start filter in a thread
    def run_filter():
        high_pass_filter.action()

    filter_thread = threading.Thread(target=run_filter)
    filter_thread.daemon = True
    filter_thread.start()

    # Collect filtered data from output store
    filtered_signal = []
    while not input_store[0].empty() or not high_pass_filter.store[0].empty():
        if not high_pass_filter.store[0].empty():
            filtered_signal.append(high_pass_filter.store[0].get())

    # Allow the filter thread to process remaining data
    filter_thread.join(timeout=1)

    # Ensure the filtered_signal is correctly populated
    if len(filtered_signal) < len(test_signal):
        print(f"Warning: Only {len(filtered_signal)} out of {len(test_signal)} samples were processed.")

    # Plot the results
    if len(filtered_signal) > 0:
        plot_signals(time[:len(filtered_signal)], test_signal[:len(filtered_signal)], np.array(filtered_signal))
    else:
        print("Error: No filtered signal data available.")

test_high_pass_filter()
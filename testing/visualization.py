import sys
from pathlib import Path

relative_path = Path("../BCI.py")
module_directory = relative_path.parent.resolve()

sys.path.insert(0, str(module_directory))

from BCI import *

# replace all file names that we're loading in with appropriate file names

headset = BCI("MuseS") # ["TP9", "AF7", "AF8", "TP10"]
headset_server, headset_server_thread = headset.launch_server()

# neutral data
headset.load_lsl_data("../EEG-recording-control-w-headphones.csv")

psd1 = PSD(headset.no_of_channels, headset.sampling_rate, headset.store[0].qsize(), headset.store)
headset1_results = psd1.compute_psd()

# disgust data
headset.load_lsl_data("../EEG-recording-disgust.csv")

psd2 = PSD(headset.no_of_channels, headset.sampling_rate, headset.store[0].qsize(), headset.store)
headset2_results = psd2.compute_psd()

# happy data
headset.load_lsl_data("../EEG-recording-happy.csv")

psd3 = PSD(headset.no_of_channels, headset.sampling_rate, headset.store[0].qsize(), headset.store)
headset3_results = psd3.compute_psd()

# surprise data
headset.load_lsl_data("../EEG-recording-surprise.csv")

psd4 = PSD(headset.no_of_channels, headset.sampling_rate, headset.store[0].qsize(), headset.store)
headset4_results = psd4.compute_psd()

fig, axs = plt.subplots(4, figsize=(20, 15))
fig.suptitle('Power Spectral Density for Each Channel')

for ax in axs.flat:
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (dB/Hz)")

for x in range(4):
    for i, (freqs, psd_values) in enumerate(eval(f"headset{x + 1}_results")):
        axs[x].semilogy(freqs, psd_values, label=f'Channel {i + 1}')

plt.legend()
plt.grid(True)
plt.savefig("psd_plot")

plt.show()

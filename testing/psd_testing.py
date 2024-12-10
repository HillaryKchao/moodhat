import sys
from pathlib import Path

relative_path = Path("../../BCI.py")
module_directory = relative_path.parent.resolve()

sys.path.insert(0, str(module_directory))

from BCI import *

headset = BCI("MuseS") # ["TP9", "AF7", "AF8", "TP10"]
headset_server, headset_server_thread = headset.launch_server()

headset.load_lsl_data("EEG_recording_2024-11-16-22.34.56.csv")

psd = PSD(headset.no_of_channels, headset.sampling_rate, headset.store[0].qsize(), headset.store)
psd.plot_psd()



# #CASE WITH PIPELINE
# import sys
# from pathlib import Path

# relative_path = Path("../../BCI.py")
# module_directory = relative_path.parent.resolve()

# sys.path.insert(0, str(module_directory))

# from BCI import *

# headset = BCI("MuseS") # ["TP9", "AF7", "AF8", "TP10"]
# headset_server, headset_server_thread = headset.launch_server()
# headset.load_lsl_data("EEG_recording_2024-11-16-22.34.56.csv")

# pipe_obj = Pipe(headset.no_of_channels, headset.sampling_rate, headset.store[0].qsize(), headset.store)


# psd = PSD(pipe_obj.no_of_channels, pipe_obj.sampling_rate, pipe_obj.store[0].qsize(), pipe_obj.store)
# psd.plot_psd()

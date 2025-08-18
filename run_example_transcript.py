import ctypes
from lattica_heal_runtime.py_wrapper import PythonToCppDispatcher
from lattica_heal_runtime.printify import print_transcript

from lattica_heal_runtime import runtime

from lattica_heal_runtime.device_interface import DeviceDispatcher
from lattica_heal_runtime.serialization import load_transcript

# Load the example transcript
transcript = load_transcript('example_transcripts/standalone_disease_prediction.json')

# Optionally print the transcript
print_transcript(transcript)

# Create a dispatcher to C++ functions
device_dispatcher = DeviceDispatcher(PythonToCppDispatcher())

# Utility function to dynamically set the number of threads for OpenMP
def _set_num_threads(num_threads=32):
    try:
        omp = ctypes.CDLL("libgomp.so.1")  # GCC OpenMP (Linux)
    except OSError:
        omp = ctypes.CDLL("libomp.dylib")  # Clang OpenMP (macOS)
    omp.omp_set_num_threads(num_threads)

# Execute the transcript with 32 threads
_set_num_threads(32)
runtime.run_transcript(
    device_dispatcher,     # Dispatcher to C++ functions
    transcript.transcript, # Set of operations to run
    verify=True)           # Verify execution results against expected results

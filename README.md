[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

# HEAL: Homomorphic Encryption Abstraction Layer

Welcome to the HEAL Runtime Repository — your integration point for executing homomorphic AI workloads on your hardware.

HEAL defines a minimal, standardized API for homomorphic encryption (FHE) operations, enabling hardware vendors to plug into real-world encrypted AI pipelines with ease.

---

## 🚀 Overview

This repository provides everything you need to integrate and test your hardware implementation against HEAL:

- 🔌 C++ function definitions and sample implementations
- 🧪 Unit tests with known input/output pairs
- 🧠 Python-based runtime that executes AI workloads from JSON transcripts
- 📊 Benchmarking hooks for performance tuning

---

## 📦 Repository Structure

```
example_impl/         # Example C++ implementations of HEAL functions (to be replaced by vendor code)
include/              # API headers: memory, arithmetic, shape, etc.
python_execution/     # Python runtime for executing HEAL transcripts
tests/                # Unit tests for each function
example_transcripts/  # Example JSON-based AI workloads
run_example_transcript.py  # Entry point to run a test workload
```

> **Note:** `example_impl/` provides a sample implementation. Hardware vendors should replace this with their own optimized implementation targeting their device.

---

## 💪 Build Instructions

### 1. Requirements

- C++17 or later
- CMake ≥ 3.14
- Python ≥ 3.11 (with development headers, i.e., python3-dev on Ubuntu/Debian, python3-devel on Fedora/Red Hat)
- Pybind11 (auto-installed)
- A C++ compiler (GCC recommended, e.g., g++. Note: clang++ may cause build issues)

### 2. Build C++ Runtime & Bindings

* Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```
Replace `{repository-url}` and `{repository-directory}` with the actual URL and directory name.

* Create a Python virtual environment:

```bash
python3 -m venv .venv
```

* Activate the virtual environment:

```bash
source .venv/bin/activate
```

* Install the Python runtime library:

```bash
pip install -e python_execution
```

This installs the Python bindings necessary for running HEAL scripts.

* Create a build directory:

```bash
mkdir build
```

* Navigate into the build directory:

```bash
cd build
```

* Generate build files using CMake:

```bash
cmake ..
```

* Compile the HEAL runtime and Python bindings:
```bash
make -j
```

This completes building the C++ runtime and Python bindings for HEAL.

---

## ▶️ Running Example Pipelines

After completing the build steps above, you can run several simulated AI models using HEAL through the provided example script.

From the root directory of the HEAL repository, execute:

```bash
python run_example_transcript.py
```

By default, this script:

1. Loads the transcript file `example_transcripts/standalone_digit_recognizer.json`.
2. Calls your C++ function implementations via the Python runtime.
3. Prints the outputs and runtime logs.

To run different examples, edit the following line inside `run_example_transcript.py`:

```python
transcript = load_transcript_from_json('example_transcripts/standalone_digit_recognizer.json')
```

Replace `'example_transcripts/standalone_digit_recognizer.json'` with the filename of your chosen example transcript.

All available example transcripts, along with their corresponding parameter files (`*_params.json`), are located in the `example_transcripts` folder.

---

## ✅ Running Unit Tests

From the build directory:

```bash
ctest --output-on-failure
```

This runs all unit tests from the `/tests` directory to verify correctness.

To run a single test executable directly and see detailed success/failure logs, execute the test binary itself:

```bash
./tests/<test_executable>
```

Replace `{test_executable}` with the actual name of your test binary.

---

## 📞 Support

Having issues? Contact us via:

- GitHub Issues (preferred)
- Slack (for verified partners)

We also welcome discussions about the HEAL specification and hardware integration strategy. If you're interested in becoming a HEAL partner, visit [lattica.ai/heal](https://www.lattica.ai/heal/) to learn more.

---

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
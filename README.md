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
example_run_transcript.py  # Entry point to run a test workload
```

> **Note:** `example_impl/` provides a sample implementation. Hardware vendors should replace this with their own optimized implementation targeting their device.

---

## 💪 Build Instructions

### 1. Requirements

- C++17 or later
- CMake ≥ 3.14
- Python ≥ 3.8
- Pybind11 (auto-installed by CMake)
- A C++ compiler (e.g., `g++` or `clang++`)

### 2. Build C++ Runtime & Bindings

```bash
mkdir build && cd build
cmake ..
make -j
```

This compiles the C++ HEAL functions and builds Python bindings used by the runtime.

---

## ▶️ Running the Example Pipeline

Run a simulated AI model using HEAL:

First, install the Python runtime:

```bash
pip install -e python_execution
```

Then run the example pipeline:

```bash
cd python_execution
python ../example_run_transcript.py
```

This will:
1. Load `standalone_matmul_simple.json`
2. Call your C++ function implementations through the Python runtime
3. Print output and runtime logs

---

## ✅ Running Unit Tests

From the build directory:

```bash
ctest --output-on-failure
```

This runs all unit tests from the `/tests` directory to verify correctness.

---

## 🧪 Testing a Custom Function

To implement and test a new function (e.g., `modmul`):

1. Add C++ code in `example_impl/modop_impl.cpp`
2. Declare the function in `include/modop.h`
3. Expose it via `py_bindings.cpp`
4. Add a unit test in `tests/test_modop.cpp`
5. Rebuild and re-run:

```bash
make -j
ctest --output-on-failure
```

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
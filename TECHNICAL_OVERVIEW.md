
## 🧠 HEAL Execution Model: Technical Overview

HEAL provides a low-level abstraction layer for executing encrypted AI workloads. It defines a **transcript-based execution model** over tensors and operations, designed to be portable, testable, and hardware-friendly.

This section outlines the core concepts hardware vendors need to understand in order to implement HEAL functions efficiently.

---

### 📐 Core Concepts

#### 1. **Tensors**

HEAL computations are expressed over **multi-dimensional tensors**. Each tensor has:

- **Shape** — A list of dimensions (e.g., `[8, 64, 64]`)
- **Data type** — For example, 64-bit integers (`int64`)
- **Memory layout** — Defined via **strides**, which allow for both contiguous and non-contiguous memory views
- **Storage location** — Indicates whether the tensor resides on the host or device

Tensors behave similarly to those in PyTorch or NumPy, but are exposed through a minimal FHE-compatible interface.

---

#### 2. **Memory Allocation & Movement**

Memory-related instructions define how and where tensor data is allocated or transferred. Examples include:

- `host_to_device` — Transfers tensors to device memory
- `device_to_host` — Moves tensors back to the host
- `empty` — Allocates a new, uninitialized tensor on device memory

These operations are crucial for efficient memory usage. Vendors should optimize these paths to align with their device architecture.

---

#### 3. **Computation Instructions**

HEAL defines a set of arithmetic and tensor manipulation operations, such as:

- `modsum`, `modmul` — Perform modular arithmetic
- `reshape`, `permute`, `repeat_interleave` — Modify or reinterpret tensor shapes
- `g_decomposition`, `ntt`, etc. — More specialized FHE-aligned operations

Each operation reads input tensors by ID, performs the computation, and writes to output tensors. 

**Note:** Some operations like `reshape` are **layout reinterpretations** — they change shape/stride metadata without modifying the underlying data in memory. Others may perform in-place or out-of-place computations depending on the instruction.

---

### 📄 Transcript-Based Execution

HEAL workloads are expressed as **instruction transcripts** in JSON format. These are emitted by a compiler and executed by the runtime.

Each instruction is a JSON object with:
- `op`: Function name (e.g., `"modsum"`)
- `args`: Input tensor IDs and constant parameters
- `output`: Output tensor ID

Example:
```json
[
  {
    "op": "host_to_device",
    "args": [...],
    "output": [...]
  },
  {
    "op": "modmul",
    "args": [...],
    "output": [...]
  },
  ...
]
```

The transcript is **linear** — it defines an ordered sequence of operations. The runtime processes them one-by-one, dispatching each call to your C++ implementation.

---

### ⚙️ The Runtime

The HEAL **runtime** is a Python-based executor that:

- Loads and parses the transcript
- Manages tensor allocation and dispatch
- Interfaces with your C++ functions via pybind11
- Tracks intermediate state and logs performance

It is located under `python_execution/` and is used for running full workloads (e.g. `example_run_transcript.py`).

---

### ♻️ CPU Fallbacks & Custom Extensions

Not all operations must be implemented in hardware. If a function is missing, the HEAL runtime will automatically fall back to a software implementation (included in the SDK) to ensure correctness and enable partial acceleration.

Additionally, **vendors can define and expose custom functions**. If provided, HEAL's compiler can generate transcripts that include these functions, enabling hardware-specific pipelines without breaking portability.

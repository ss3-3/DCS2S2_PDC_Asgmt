# Matrix Inversion Benchmark Suite

A comprehensive C++ benchmarking tool that compares two popular matrix inversion algorithms (Gauss-Jordan and LU Decomposition) using both serial and parallel (OpenMP) implementations.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Compilation](#compilation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Customization](#customization)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This program benchmarks matrix inversion using two algorithms:

1. **Gauss-Jordan Method** - Simple, highly parallelizable
2. **LU Decomposition** - Numerically stable, more complex

Each algorithm is tested with:
- **Serial implementation** (single-threaded baseline)
- **OpenMP implementation** (multi-threaded parallel)

The program measures execution time, speedup, and parallel efficiency for matrices of various sizes.

---

## Features

- **Two Dataset Modes:**
  - **Static Dataset** - Test with your own predefined matrix
  - **Dynamic Dataset** - Generate random invertible matrices (10×10 to 2000×2000)

- **Thread Configuration:**
  - Automatic detection of available CPU threads
  - Configurable thread count for OpenMP algorithms
  - Per-region thread control without global state pollution

- **Comprehensive Metrics:**
  - Execution time (milliseconds)
  - Speedup ratio (Serial vs Parallel)
  - Parallel efficiency percentage
  - Matrix invertibility verification (determinant check)
  - Result accuracy verification (A × A⁻¹ = I)

- **Safety Features:**
  - Input validation
  - Singular matrix detection
  - User confirmation for large matrices
  - Progress indicators for long operations

---

## Requirements

### Software Requirements
- **C++ Compiler** with C++11 support or higher
  - GCC 4.9+ (recommended)
  - Clang 3.4+
  - MSVC 2015+
  
- **OpenMP Library** (usually included with compiler)
  - GCC: Built-in (use `-fopenmp` flag)
  - Clang: May require separate installation
  - MSVC: Built-in (use `/openmp` flag)

### Hardware Requirements
- **Minimum:** 2 CPU cores, 4GB RAM
- **Recommended for large matrices (1000×1000+):**
  - 4+ CPU cores
  - 8GB+ RAM
  - 64-bit system

---

## Compilation

### Linux / macOS (GCC)
```bash
g++ -o matrix_benchmark matrix_inversion.cpp -fopenmp -O3 -std=c++11
```

### Linux / macOS (Clang)
```bash
clang++ -o matrix_benchmark matrix_inversion.cpp -fopenmp -O3 -std=c++11
```

### Windows (MSVC)
```cmd
cl /EHsc /openmp /O2 /std:c++14 matrix_inversion.cpp /Fe:matrix_benchmark.exe
```

### Compilation Flags Explained:
- `-fopenmp` / `/openmp` - Enable OpenMP support
- `-O3` / `/O2` - Enable optimizations
- `-std=c++11` - Use C++11 standard

---

## Usage

### Running the Program
```bash
./matrix_benchmark
```

### Main Menu Options

```
1. Static Dataset (Predefined Matrix)
2. Dynamic Dataset (Generate Random Matrix)
3. Configure Thread Count
0. Exit
```

### Workflow Example

#### Option 1: Static Dataset
1. Select **Option 1** from main menu
2. Program loads predefined matrix from source code
3. Validates matrix invertibility (checks determinant)
4. Runs both algorithms (Serial + OpenMP)
5. Displays timing results and verification

#### Option 2: Dynamic Dataset
1. Select **Option 2** from main menu
2. Choose matrix size:
   - **10×10** - Quick test (~milliseconds)
   - **100×100** - Fast (~seconds)
   - **500×500** - Moderate (~10-30 seconds)
   - **1000×1000** - Slow (~1-5 minutes)
   - **2000×2000** - Very slow (~10-30 minutes)
3. Program generates random invertible matrix
4. Runs benchmarks and displays results

#### Option 3: Thread Configuration
1. Select **Option 3** from main menu
2. Choose:
   - **Option 0** - Use maximum available threads
   - **Option 1** - Set custom thread count (1 to max)
3. Setting applies to all subsequent benchmarks

---

## 📊 Algorithm Details

### Gauss-Jordan Method
**How it works:**
- Augments matrix A with identity matrix I → [A|I]
- Applies row operations to transform [A|I] → [I|A⁻¹]
- Simple and highly parallelizable

**Parallelization:**
- Pivot search across rows
- Row normalization
- Row elimination operations

**Advantages:**
- Easy to implement
- Good parallel scalability
- Direct inversion

**Disadvantages:**
- Less numerically stable
- More operations than LU

---

### LU Decomposition
**How it works:**
- Decomposes A into Lower (L) and Upper (U) triangular matrices
- PA = LU (with permutation matrix P)
- Solves LU·X = I to find A⁻¹

**Parallelization:**
- Pivot search
- Gaussian elimination steps

**Advantages:**
- Numerically stable
- Efficient for multiple inversions
- Reusable decomposition

**Disadvantages:**
- More complex implementation
- Limited parallelization opportunities
- Requires additional solving step

---

## Customization

### Changing the Static Dataset

Locate this section in the source code (around line 215):

```cpp
// ============================================================================
// STATIC DATASET CONFIGURATION
// ============================================================================

const int N_STATIC = 5;  // <-- Change matrix size here

// <-- Replace with your matrix values
double static_data[N_STATIC][N_STATIC] = {
    {4.0,  2.0, -1.0,  3.0,  1.0},
    {1.0,  5.0,  2.0, -1.0,  2.0},
    {2.0,  1.0,  6.0,  1.0, -2.0},
    {-1.0, 2.0,  1.0,  7.0,  3.0},
    {3.0, -1.0,  2.0,  1.0,  8.0}
};
```

**Steps to customize:**
1. Change `N_STATIC` to your matrix dimension
2. Replace `static_data` array with your values
3. Recompile the program
4. Run and select "Static Dataset" option

**Important:** Make sure your matrix is invertible (determinant ≠ 0)!

---

## Performance Metrics

### Output Example
```
COMPARISON SUMMARY FOR 500x500 (Using 8 threads)
================================================================================
Algorithm              Serial (ms)     OpenMP (ms)     Speedup    Efficiency (%)
--------------------------------------------------------------------------------
Gauss-Jordan          8523.45         1234.67         6.90x      86.25%
LU Decomposition      6789.12         1456.89         4.66x      58.25%
================================================================================

>>> FASTEST METHOD: Gauss-Jordan OpenMP (1234.67 ms)
```

### Metrics Explained

**Execution Time**
- Time taken to complete inversion (milliseconds)
- Lower is better

**Speedup**
```
Speedup = Serial Time / Parallel Time
```
- Ideal speedup = Number of threads
- Example: 8 threads → ideal speedup = 8x

**Efficiency**
```
Efficiency = (Speedup / Number of Threads) × 100%
```
- Measures how well threads are utilized
- 100% = perfect scaling
- 70%+ = good scaling
- <50% = poor scaling (overhead dominates)

---

## Troubleshooting

### Issue: "Matrix is singular"
**Cause:** Determinant is zero or very close to zero  
**Solution:** 
- For static dataset: Ensure matrix rows/columns are linearly independent
- For dynamic dataset: Program automatically regenerates - if persistent, file a bug report

---

### Issue: OpenMP not found during compilation
**Cause:** OpenMP library not installed or not in compiler path

**Linux (GCC):**
```bash
sudo apt-get install libomp-dev  # Ubuntu/Debian
sudo yum install libomp-devel    # CentOS/RHEL
```

**macOS:**
```bash
brew install libomp
# Then compile with: clang++ -Xpreprocessor -fopenmp -lomp ...
```

**Windows (MSVC):** OpenMP is included by default

---

### Issue: "Verification FAILED"
**Cause:** Numerical errors in computation

**Solutions:**
- Increase tolerance in `verify_inverse()` function (line ~287)
- Check if matrix is ill-conditioned (very large or very small determinant)
- Try a different matrix

---

### Issue: Program crashes with large matrices
**Cause:** Insufficient memory

**Solutions:**
- Close other applications
- Use smaller matrix size
- Increase system swap/virtual memory
- Use 64-bit compilation

---

### Issue: Very slow performance
**Possible causes:**
- Running on low-end hardware
- Too few threads allocated
- System under heavy load
- Debug build instead of optimized build

**Solutions:**
- Verify optimization flags (`-O3` or `/O2`)
- Check thread configuration (Option 3 in menu)
- Close background applications
- Use smaller matrices for testing

---

## Expected Performance

### Typical Execution Times (8-core CPU, 3.0 GHz)

| Matrix Size | Serial GJ | OpenMP GJ | Serial LU | OpenMP LU |
|-------------|-----------|-----------|-----------|-----------|
| 10×10       | <1 ms     | <1 ms     | <1 ms     | <1 ms     |
| 100×100     | 15 ms     | 3 ms      | 12 ms     | 4 ms      |
| 500×500     | 8500 ms   | 1200 ms   | 6800 ms   | 1400 ms   |
| 1000×1000   | 68 sec    | 9 sec     | 54 sec    | 12 sec    |
| 2000×2000   | 9 min     | 75 sec    | 7 min     | 95 sec    |

*Times are approximate and vary by hardware*

---

## Contributing

### Adding New Algorithms
1. Implement serial version
2. Implement OpenMP version
3. Add to `run_single_benchmark()` function
4. Update comparison table output

### Improving Existing Code
- Optimize parallel regions
- Add error handling
- Improve numerical stability
- Add more matrix validation

---

## License

This code is provided for educational purposes. Feel free to modify and distribute.

---

## Author Notes

**For Students/Researchers:**
- This code is designed for learning parallel programming concepts
- Experiment with different thread counts and matrix sizes
- Compare algorithmic complexity vs parallel efficiency
- Perfect for parallel computing assignments

**For Developers:**
- Code uses modern C++11 features
- OpenMP directives are clearly commented
- Easy to extend with new algorithms
- Production use may require additional error handling and optimizations

---

## Further Reading

- [OpenMP Official Documentation](https://www.openmp.org/)
- [Matrix Inversion Algorithms](https://en.wikipedia.org/wiki/Invertible_matrix#Methods_of_matrix_inversion)
- [Parallel Algorithm Design](https://en.wikipedia.org/wiki/Parallel_algorithm)
- [Numerical Linear Algebra](https://en.wikipedia.org/wiki/Numerical_linear_algebra)

---

**Last Updated:** December 2025  
**Version:** 1.0  
**Language:** C++11 with OpenMP

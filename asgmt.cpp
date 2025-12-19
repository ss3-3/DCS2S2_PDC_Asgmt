#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <omp.h>
using namespace std;
using namespace std::chrono;

// ============================================================================
// INPUT VALIDATION FUNCTIONS
// ============================================================================

bool get_valid_int(int& value, int min_val, int max_val) {
    if (!(cin >> value)) {
        cin.clear();
        cin.ignore(10000, '\n');
        return false;
    }
    cin.ignore(10000, '\n');
    return (value >= min_val && value <= max_val);
}

bool get_valid_char(char& value) {
    if (!(cin >> value)) {
        cin.clear();
        cin.ignore(10000, '\n');
        return false;
    }
    cin.ignore(10000, '\n');
    return true;
}

// ============================================================================
// GLOBAL THREAD CONFIGURATION
// ============================================================================

// ============================================================================
// THREAD CONFIGURATION FUNCTIONS
// ============================================================================

int get_max_threads() {
    return omp_get_max_threads();
}

int get_current_thread_setting(int num_threads) {
    return (num_threads == 0) ? get_max_threads() : num_threads;
}

void display_thread_menu(int num_threads) {
    int max_threads = get_max_threads();
    int current = get_current_thread_setting(num_threads);

    cout << "\n" << string(70, '=') << "\n";
    cout << "THREAD CONFIGURATION\n";
    cout << string(70, '=') << "\n";
    cout << "Maximum available threads: " << max_threads << "\n";
    cout << "Current setting: " << current << " thread(s)\n";
    cout << string(70, '-') << "\n";
    cout << "  0. Use Maximum Threads (Default: " << max_threads << ")\n";
    cout << "  1. Set Custom Thread Count\n";
    cout << "  2. Back to Main Menu\n";
    cout << string(70, '=') << "\n";
}

void handle_thread_configuration(int& num_threads) {
    int max_threads = get_max_threads();

    while (true) {
        display_thread_menu(num_threads);

        cout << "\nEnter your choice (0-2): ";
        int choice;

        if (!get_valid_int(choice, 0, 2)) {
            cout << "\n[ERROR] Invalid input! Please enter a number between 0 and 2.\n";
            cout << "Press Enter to continue...";
            cin.get();
            continue;
        }

        if (choice == 0) {
            num_threads = 0;  // Use maximum
            cout << "\n[INFO] Thread count set to MAXIMUM (" << max_threads << " threads)\n";
            cout << "Press Enter to continue...";
            cin.get();
            break;
        }
        else if (choice == 1) {
            cout << "\nEnter number of threads (1-" << max_threads << "): ";
            int num;

            if (!get_valid_int(num, 1, max_threads)) {
                cout << "\n[ERROR] Invalid input! Please enter a number between 1 and " << max_threads << "\n";
                cout << "Press Enter to continue...";
                cin.get();
                continue;
            }

            num_threads = num;
            cout << "\n[INFO] Thread count set to " << num << "\n";
            cout << "Press Enter to continue...";
            cin.get();
            break;
        }
        else if (choice == 2) {
            break;
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

string int_to_string(int num) {
    stringstream ss;
    ss << num;
    return ss.str();
}

void print_matrix(const vector<vector<double>>& M, int rows, int cols, const string& title) {
    cout << "\n" << title << "\n";
    cout << string(50, '=') << "\n";

    if (rows > 10 || cols > 10) {
        cout << "[Matrix too large to display (Size: " << rows << "x" << cols << ") ]\n";
        cout << "Showing first 5x5 corner:\n";
        int display_rows = min(5, rows);
        int display_cols = min(5, cols);
        for (int i = 0; i < display_rows; ++i) {
            for (int j = 0; j < display_cols; ++j) {
                cout << setw(12) << fixed << setprecision(4) << M[i][j];
            }
            cout << " ...\n";
        }
        cout << "...\n";
    }
    else {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << setw(12) << fixed << setprecision(4) << M[i][j];
            }
            cout << "\n";
        }
    }
    cout << string(50, '=') << "\n";
}

// Calculate determinant using LU decomposition
double calculate_determinant(const vector<vector<double>>& A, int N) {
    vector<vector<double>> U = A;
    double det = 1.0;
    int sign = 1;

    for (int k = 0; k < N; ++k) {
        // Find pivot
        int pivot_row = k;
        for (int i = k + 1; i < N; ++i) {
            if (abs(U[i][k]) > abs(U[pivot_row][k])) {
                pivot_row = i;
            }
        }

        if (abs(U[pivot_row][k]) < 1e-10) {
            return 0.0; // Singular matrix
        }

        if (pivot_row != k) {
            swap(U[k], U[pivot_row]);
            sign *= -1;
        }

        det *= U[k][k];

        // Eliminate
        for (int i = k + 1; i < N; ++i) {
            double factor = U[i][k] / U[k][k];
            for (int j = k; j < N; ++j) {
                U[i][j] -= factor * U[k][j];
            }
        }
    }

    return sign * det;
}

void generate_random_matrix(vector<vector<double>>& A, int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-10.0, 10.0);

    for (int i = 0; i < N; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                A[i][j] = dis(gen);
                row_sum += abs(A[i][j]);
            }
        }
        // Make diagonally dominant to ensure invertibility
        A[i][i] = row_sum + abs(dis(gen)) + 1.0;
    }
}

// ============================================================================
// STATIC DATASET CONFIGURATION
// ============================================================================
// ** TUTOR: REPLACE THIS SECTION WITH YOUR OWN DATASET **
// Instructions:
// 1. Change N_STATIC to match your matrix size
// 2. Replace the static_data array with your values

const int N_STATIC = 5;  // <-- CHANGE THIS to your matrix size

// <-- REPLACE THIS ARRAY with your dataset
double static_data[N_STATIC][N_STATIC] = {
    {4.0,  2.0, -1.0,  3.0,  1.0},
    {1.0,  5.0,  2.0, -1.0,  2.0},
    {2.0,  1.0,  6.0,  1.0, -2.0},
    {-1.0, 2.0,  1.0,  7.0,  3.0},
    {3.0, -1.0,  2.0,  1.0,  8.0}
};

// Function to validate and load static dataset
bool validate_static_data() {
    cout << "[INFO] Validating static dataset...\n";

    // Check for NaN or Inf values
    int nan_count = 0;
    int inf_count = 0;
    int zero_count = 0;
    int negative_count = 0;
    int float_count = 0;

    for (int i = 0; i < N_STATIC; ++i) {
        for (int j = 0; j < N_STATIC; ++j) {
            double val = static_data[i][j];

            if (isnan(val)) {
                nan_count++;
                cout << "[ERROR] NaN detected at position [" << i << "][" << j << "]\n";
            }
            if (isinf(val)) {
                inf_count++;
                cout << "[ERROR] Infinity detected at position [" << i << "][" << j << "]\n";
            }
            if (val == 0.0) {
                zero_count++;
            }
            if (val < 0.0) {
                negative_count++;
            }
            if (val != floor(val)) {
                float_count++;
            }
        }
    }

    // Display statistics
    cout << "\n[VALIDATION REPORT]\n";
    cout << string(50, '-') << "\n";
    cout << "Matrix Size:        " << N_STATIC << "x" << N_STATIC << "\n";
    cout << "Total Elements:     " << (N_STATIC * N_STATIC) << "\n";
    cout << "Zero values:        " << zero_count << "\n";
    cout << "Negative values:    " << negative_count << "\n";
    cout << "Floating-point:     " << float_count << "\n";
    cout << "NaN values:         " << nan_count << " " << (nan_count > 0 ? "INVALID" : "VALID") << "\n";
    cout << "Infinity values:    " << inf_count << " " << (inf_count > 0 ? "INVALID" : "VALID") << "\n";
    cout << string(50, '-') << "\n";

    if (nan_count > 0 || inf_count > 0) {
        cout << "\n[ERROR] Dataset contains invalid values (NaN or Infinity)!\n";
        cout << "[ERROR] Please fix the static_data array and recompile.\n";
        return false;
    }

    cout << "[INFO] Dataset validation PASSED \n";
    return true;
}

void load_static_matrix(vector<vector<double>>& A, int& N) {
    N = N_STATIC;
    A.assign(N, vector<double>(N));

    // Load the static data
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = static_data[i][j];
        }
    }
}

bool verify_inverse(const vector<vector<double>>& A, const vector<vector<double>>& A_inv, int N) {
    int verify_size = min(N, 5);
    vector<vector<double>> result(verify_size, vector<double>(verify_size, 0.0));
    bool is_identity = true;

    for (int i = 0; i < verify_size; ++i) {
        for (int j = 0; j < verify_size; ++j) {
            for (int k = 0; k < N; ++k) {
                result[i][j] += A[i][k] * A_inv[k][j];
            }
        }
    }

    for (int i = 0; i < verify_size; ++i) {
        for (int j = 0; j < verify_size; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (abs(result[i][j] - expected) > 1e-3) {
                is_identity = false;
            }
        }
    }

    return is_identity;
}

// ============================================================================
// GAUSS-JORDAN IMPLEMENTATIONS
// ============================================================================

bool gauss_jordan_serial(vector<vector<double>>& M, int N, double& exec_time) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        if (N >= 1000 && i % (N / 10) == 0) {
            cout << "Progress: " << (i * 100 / N) << "%\r" << flush;
        }

        // Partial pivoting
        int pivot_row = i;
        for (int k = i + 1; k < N; ++k) {
            if (abs(M[k][i]) > abs(M[pivot_row][i])) {
                pivot_row = k;
            }
        }

        if (abs(M[pivot_row][i]) < 1e-10) {
            cout << "\n[ERROR] Matrix is singular!\n";
            return false;
        }

        if (pivot_row != i) {
            swap(M[i], M[pivot_row]);
        }

        // Normalize pivot row
        double pivot_val = M[i][i];
        for (int j = 0; j < 2 * N; ++j) {
            M[i][j] /= pivot_val;
        }

        // Eliminate all other rows
        for (int k = 0; k < N; ++k) {
            if (k != i) {
                double factor = M[k][i];
                for (int j = 0; j < 2 * N; ++j) {
                    M[k][j] -= factor * M[i][j];
                }
            }
        }
    }

    if (N >= 1000) cout << "Progress: 100%\n";

    auto end = high_resolution_clock::now();
    exec_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    return true;
}

//openmp version for gauss jordan elimination
bool gauss_jordan_openmp(vector<vector<double>>& M, int N, double& exec_time, int num_threads) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        // Parallel pivot search
        int pivot_row = i;
        double pivot_max = abs(M[i][i]);

#pragma omp parallel num_threads(num_threads)
        {
            int local_row = pivot_row;
            double local_max = pivot_max;

#pragma omp for nowait
            for (int k = i + 1; k < N; ++k) {
                double val = abs(M[k][i]);
                if (val > local_max) {
                    local_max = val;
                    local_row = k;
                }
            }

#pragma omp critical
            {
                if (local_max > pivot_max) {
                    pivot_max = local_max;
                    pivot_row = local_row;
                }
            }
        }

        if (pivot_max < 1e-12) return false;

        if (pivot_row != i) swap(M[i], M[pivot_row]);

        // Parallel normalization
        double pivot_val = M[i][i];
#pragma omp parallel for num_threads(num_threads)
        for (int j = 0; j < 2 * N; ++j) {
            M[i][j] /= pivot_val;
        }

        // Parallel elimination
#pragma omp parallel for num_threads(num_threads)
        for (int k = 0; k < N; ++k) {
            if (k != i) {
                double factor = M[k][i];
                for (int j = 0; j < 2 * N; ++j) {
                    M[k][j] -= factor * M[i][j];
                }
            }
        }
    }

    auto end = high_resolution_clock::now();
    exec_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    return true;
}

// ============================================================================
// LU DECOMPOSITION IMPLEMENTATIONS
// ============================================================================

bool lu_decomposition_serial(const vector<vector<double>>& A, vector<vector<double>>& L,
    vector<vector<double>>& U, vector<int>& P, int N, double& exec_time) {
    auto start = high_resolution_clock::now();

    // Initialize
    L.assign(N, vector<double>(N, 0.0));
    U = A;
    P.resize(N);
    for (int i = 0; i < N; ++i) {
        L[i][i] = 1.0;
        P[i] = i;
    }

    for (int k = 0; k < N; ++k) {
        if (N >= 1000 && k % (N / 10) == 0) {
            cout << "Progress: " << (k * 100 / N) << "%\r" << flush;
        }

        // Partial pivoting
        int pivot_row = k;
        for (int i = k + 1; i < N; ++i) {
            if (abs(U[i][k]) > abs(U[pivot_row][k])) {
                pivot_row = i;
            }
        }

        if (abs(U[pivot_row][k]) < 1e-10) {
            cout << "\n[ERROR] Matrix is singular!\n";
            return false;
        }

        if (pivot_row != k) {
            swap(U[k], U[pivot_row]);
            swap(P[k], P[pivot_row]);
            for (int j = 0; j < k; ++j) {
                swap(L[k][j], L[pivot_row][j]);
            }
        }

        // Gaussian elimination
        for (int i = k + 1; i < N; ++i) {
            L[i][k] = U[i][k] / U[k][k];
            for (int j = k; j < N; ++j) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    if (N >= 1000) cout << "Progress: 100%\n";

    auto end = high_resolution_clock::now();
    exec_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    return true;
}

bool lu_decomposition_openmp(const vector<vector<double>>& A, vector<vector<double>>& L,
    vector<vector<double>>& U, vector<int>& P, int N,
    double& exec_time, int num_threads) {
    auto start = high_resolution_clock::now();

    L.assign(N, vector<double>(N, 0.0));
    U = A;
    P.resize(N);
    for (int i = 0; i < N; ++i) {
        L[i][i] = 1.0;
        P[i] = i;
    }

    for (int k = 0; k < N; ++k) {
        // Parallel pivot search
        int pivot_row = k;
        double pivot_max = abs(U[k][k]);

#pragma omp parallel num_threads(num_threads)
        {
            int local_row = pivot_row;
            double local_max = pivot_max;

#pragma omp for nowait
            for (int i = k + 1; i < N; ++i) {
                double val = abs(U[i][k]);
                if (val > local_max) {
                    local_max = val;
                    local_row = i;
                }
            }

#pragma omp critical
            {
                if (local_max > pivot_max) {
                    pivot_max = local_max;
                    pivot_row = local_row;
                }
            }
        }

        if (pivot_max < 1e-12) return false;

        if (pivot_row != k) {
            swap(U[k], U[pivot_row]);
            swap(P[k], P[pivot_row]);
            for (int j = 0; j < k; ++j) {
                swap(L[k][j], L[pivot_row][j]);
            }
        }

        // Parallel elimination
#pragma omp parallel for num_threads(num_threads)
        for (int i = k + 1; i < N; ++i) {
            L[i][k] = U[i][k] / U[k][k];
            for (int j = k; j < N; ++j) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    auto end = high_resolution_clock::now();
    exec_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    return true;
}

// Solve using LU decomposition: A*X = I, so X = A^-1
void solve_with_lu(const vector<vector<double>>& L, const vector<vector<double>>& U,
    const vector<int>& P, vector<vector<double>>& A_inv, int N) {
    A_inv.assign(N, vector<double>(N, 0.0));

    // Solve for each column of the inverse
    for (int col = 0; col < N; ++col) {
        vector<double> b(N, 0.0);
        b[col] = 1.0;

        // Apply permutation
        vector<double> pb(N);
        for (int i = 0; i < N; ++i) {
            pb[i] = b[P[i]];
        }

        // Forward substitution: L*y = pb
        vector<double> y(N);
        for (int i = 0; i < N; ++i) {
            y[i] = pb[i];
            for (int j = 0; j < i; ++j) {
                y[i] -= L[i][j] * y[j];
            }
        }

        // Back substitution: U*x = y
        vector<double> x(N);
        for (int i = N - 1; i >= 0; --i) {
            x[i] = y[i];
            for (int j = i + 1; j < N; ++j) {
                x[i] -= U[i][j] * x[j];
            }
            x[i] /= U[i][i];
        }

        // Store column in result
        for (int i = 0; i < N; ++i) {
            A_inv[i][col] = x[i];
        }
    }
}

// ============================================================================
// BENCHMARK FUNCTIONS
// ============================================================================

void run_single_benchmark(int size, const vector<vector<double>>& A, int num_threads) {
    cout << "\n" << string(70, '#') << "\n";
    cout << "##  TESTING MATRIX SIZE: " << size << "x" << size;
    size_t padding = 70 - 26 - int_to_string(size).length() * 2 - 1;
    for (int p = 0; p < padding; ++p) cout << ' ';
    cout << "##\n";
    cout << string(70, '#') << "\n";

    if (size <= 10) {
        print_matrix(A, size, size, "Matrix A");
    }
    else {
        cout << "[INFO] Matrix loaded successfully (too large to display)\n";
    }

    // Get actual thread count
    int actual_threads = get_current_thread_setting(num_threads);
    cout << "\n[INFO] Using " << actual_threads << " thread(s) for OpenMP algorithms\n";

    // ========== GAUSS-JORDAN ==========
    cout << "\n" << string(70, '=') << "\n";
    cout << "GAUSS-JORDAN METHOD\n";
    cout << string(70, '=') << "\n";

    // Serial
    vector<vector<double>> M_serial(size, vector<double>(2 * size, 0.0));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) M_serial[i][j] = A[i][j];
        M_serial[i][i + size] = 1.0;
    }

    double gj_serial_time;
    cout << "[Serial] Running Gauss-Jordan (1 thread)...\n";
    bool gj_success = gauss_jordan_serial(M_serial, size, gj_serial_time);

    if (!gj_success) {
        cout << "[ERROR] Serial Gauss-Jordan failed!\n";
        return;
    }

    vector<vector<double>> A_inv_gj_serial(size, vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A_inv_gj_serial[i][j] = M_serial[i][j + size];
        }
    }
    bool verified = verify_inverse(A, A_inv_gj_serial, size);
    cout << "[Serial] Execution Time: " << fixed << setprecision(2) << gj_serial_time << " ms\n";
    cout << "[Serial] Verification: " << (verified ? "PASSED " : "FAILED ") << "\n";

    // OpenMP
    vector<vector<double>> M_openmp(size, vector<double>(2 * size, 0.0));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) M_openmp[i][j] = A[i][j];
        M_openmp[i][i + size] = 1.0;
    }

    double gj_openmp_time;
    cout << "[OpenMP] Running Gauss-Jordan with " << actual_threads << " threads...\n";
    gauss_jordan_openmp(M_openmp, size, gj_openmp_time, actual_threads);

    vector<vector<double>> A_inv_gj_openmp(size, vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A_inv_gj_openmp[i][j] = M_openmp[i][j + size];
        }
    }
    verified = verify_inverse(A, A_inv_gj_openmp, size);
    cout << "[OpenMP] Execution Time: " << fixed << setprecision(2) << gj_openmp_time << " ms\n";
    cout << "[OpenMP] Verification: " << (verified ? "PASSED " : "FAILED ") << "\n";

    double gj_speedup = gj_serial_time / gj_openmp_time;
    double gj_efficiency = (gj_speedup / actual_threads) * 100;
    cout << "[OpenMP] Speedup: " << fixed << setprecision(2) << gj_speedup << "x\n";
    cout << "[OpenMP] Efficiency: " << fixed << setprecision(2) << gj_efficiency << "%\n";

    // ========== LU DECOMPOSITION ==========
    cout << "\n" << string(70, '=') << "\n";
    cout << "LU DECOMPOSITION METHOD\n";
    cout << string(70, '=') << "\n";

    // Serial
    vector<vector<double>> L, U;
    vector<int> P;
    double lu_serial_time;
    cout << "[Serial] Running LU Decomposition (1 thread)...\n";
    bool lu_success = lu_decomposition_serial(A, L, U, P, size, lu_serial_time);

    if (!lu_success) {
        cout << "[ERROR] Serial LU Decomposition failed!\n";
        return;
    }

    vector<vector<double>> A_inv_lu_serial;
    solve_with_lu(L, U, P, A_inv_lu_serial, size);
    verified = verify_inverse(A, A_inv_lu_serial, size);
    cout << "[Serial] Execution Time: " << fixed << setprecision(2) << lu_serial_time << " ms\n";
    cout << "[Serial] Verification: " << (verified ? "PASSED " : "FAILED ") << "\n";

    // OpenMP
    vector<vector<double>> L_omp, U_omp;
    vector<int> P_omp;
    double lu_openmp_time;
    cout << "[OpenMP] Running LU Decomposition with " << actual_threads << " threads...\n";
    lu_decomposition_openmp(A, L_omp, U_omp, P_omp, size, lu_openmp_time, actual_threads);

    vector<vector<double>> A_inv_lu_openmp;
    solve_with_lu(L_omp, U_omp, P_omp, A_inv_lu_openmp, size);
    verified = verify_inverse(A, A_inv_lu_openmp, size);
    cout << "[OpenMP] Execution Time: " << fixed << setprecision(2) << lu_openmp_time << " ms\n";
    cout << "[OpenMP] Verification: " << (verified ? "PASSED " : "FAILED ") << "\n";

    double lu_speedup = lu_serial_time / lu_openmp_time;
    double lu_efficiency = (lu_speedup / actual_threads) * 100;
    cout << "[OpenMP] Speedup: " << fixed << setprecision(2) << lu_speedup << "x\n";
    cout << "[OpenMP] Efficiency: " << fixed << setprecision(2) << lu_efficiency << "%\n";

    // ========== COMPARISON ==========
    cout << "\n" << string(80, '=') << "\n";
    cout << "COMPARISON SUMMARY FOR " << size << "x" << size << " (Using " << actual_threads << " threads)\n";
    cout << string(80, '=') << "\n";
    cout << left << setw(22) << "Algorithm"
        << setw(16) << "Serial (ms)"
        << setw(16) << "OpenMP (ms)"
        << setw(14) << "Speedup"
        << setw(16) << "Efficiency (%)" << "\n";
    cout << string(80, '-') << "\n";

    cout << left << setw(22) << "Gauss-Jordan"
        << setw(16) << fixed << setprecision(2) << gj_serial_time
        << setw(16) << gj_openmp_time
        << setw(14) << (to_string(gj_speedup).substr(0, 4) + "x")
        << setw(16) << gj_efficiency << "\n";

    cout << left << setw(22) << "LU Decomposition"
        << setw(16) << fixed << setprecision(2) << lu_serial_time
        << setw(16) << lu_openmp_time
        << setw(14) << (to_string(lu_speedup).substr(0, 4) + "x")
        << setw(16) << lu_efficiency << "\n";
    cout << string(80, '=') << "\n";

    // Winner
    vector<pair<string, double>> times = {
        {"Gauss-Jordan Serial", gj_serial_time},
        {"Gauss-Jordan OpenMP", gj_openmp_time},
        {"LU Decomposition Serial", lu_serial_time},
        {"LU Decomposition OpenMP", lu_openmp_time}
    };
    auto best = min_element(times.begin(), times.end(),
        [](const pair<string, double>& a, const pair<string, double>& b) {
            return a.second < b.second;
        });

    cout << "\n>>> FASTEST METHOD: " << best->first
        << " (" << fixed << setprecision(2) << best->second << " ms)\n";
    cout << string(70, '=') << "\n";
}

// ============================================================================
// MENU SYSTEM
// ============================================================================

void display_main_menu(int num_threads) {
    int current_threads = get_current_thread_setting(num_threads);
    int max_threads = get_max_threads();

    cout << "\n" << string(70, '=') << "\n";
    cout << "MATRIX INVERSION BENCHMARK - MAIN MENU\n";
    cout << string(70, '=') << "\n";
    cout << "Current Thread Setting: " << current_threads << "/" << max_threads << " threads\n";
    cout << string(70, '-') << "\n";
    cout << "  1. Static Dataset (Predefined Matrix)\n";
    cout << "  2. Dynamic Dataset (Generate Random Matrix)\n";
    cout << "  3. Configure Thread Count\n";
    cout << "  0. Exit\n";
    cout << string(70, '=') << "\n";
}

void display_dynamic_menu() {
    cout << "\n" << string(70, '=') << "\n";
    cout << "DYNAMIC DATASET - MATRIX SIZE SELECTION\n";
    cout << string(70, '=') << "\n";
    cout << "  1. 10x10       (Quick test, ~milliseconds)\n";
    cout << "  2. 100x100     (Fast, ~seconds)\n";
    cout << "  3. 500x500     (Moderate, ~10-30 seconds)\n";
    cout << "  4. 1000x1000   (Slow, ~1-5 minutes)\n";
    cout << "  5. 2000x2000   (Very slow, ~10-30 minutes)\n";
    cout << "  0. Back to Main Menu\n";
    cout << string(70, '=') << "\n";
}

void handle_static_dataset(int num_threads) {
    cout << "\n[INFO] Loading static dataset...\n";

    // Validate the static data first
    if (!validate_static_data()) {
        cout << "\n[ABORT] Static dataset validation failed!\n";
        return;
    }

    vector<vector<double>> A;
    int N;
    load_static_matrix(A, N);

    cout << "\n[INFO] Static matrix loaded: " << N << "x" << N << "\n";

    // Display matrix (with size limits)
    if (N <= 10) {
        print_matrix(A, N, N, "Static Matrix");
    }
    else {
        cout << "[INFO] Matrix is large (" << N << "x" << N << "), showing first 5x5:\n";
        print_matrix(A, N, N, "Static Matrix (Preview)");
    }

    // Check determinant
    cout << "\n[INFO] Checking if matrix is invertible...\n";
    double det = calculate_determinant(A, N);
    cout << "[INFO] Determinant = " << scientific << setprecision(6) << det << "\n";

    if (abs(det) < 1e-10) {
        cout << "\n" << string(70, '!') << "\n";
        cout << "!! [ERROR] Matrix is SINGULAR (determinant approximately equal to 0)              !!\n";
        cout << "!! This matrix has NO INVERSE and CANNOT be inverted!       !!\n";
        cout << string(70, '!') << "\n";
        cout << "\n[EXPLANATION]\n";
        cout << "A singular matrix means its rows or columns are linearly dependent.\n";
        cout << "This happens when:\n";
        cout << "  - One row is a multiple of another row\n";
        cout << "  - One column is a combination of other columns\n";
        cout << "  - The matrix represents a transformation that loses information\n";
        cout << "\nPlease provide a different matrix with non-zero determinant.\n";
        return;
    }

    cout << "[INFO] Matrix is INVERTIBLE (determinant not equal to 0)\n";

    // Confirm before running (especially for large matrices)
    if (N >= 500) {
        cout << "\n[WARNING] Matrix size is " << N << "x" << N << " - this may take several minutes!\n";
        cout << "Continue with inversion? (y/n): ";
        char confirm;

        if (!get_valid_char(confirm)) {
            cout << "\n[ERROR] Invalid input. Cancelled.\n";
            return;
        }

        if (confirm != 'y' && confirm != 'Y') {
            cout << "[INFO] Cancelled.\n";
            return;
        }
    }

    cout << "\n[INFO] Proceeding with matrix inversion benchmarks...\n";

    run_single_benchmark(N, A, num_threads);
}

void handle_dynamic_dataset(int num_threads) {
    vector<int> available_sizes = { 10, 100, 500, 1000, 2000 };

    while (true) {
        display_dynamic_menu();

        cout << "\nEnter your choice (0-5): ";
        int choice;

        if (!get_valid_int(choice, 0, 5)) {
            cout << "\n[ERROR] Invalid input! Please enter a number between 0 and 5.\n";
            cout << "Press Enter to continue...";
            cin.get();
            continue;
        }

        if (choice == 0) {
            cout << "\n[INFO] Returning to main menu...\n";
            break;
        }
        else if (choice >= 1 && choice <= 5) {
            int size = available_sizes[choice - 1];

            // Confirm for large matrices
            if (size >= 2000) {
                cout << "\n[WARNING] " << size << "x" << size
                    << " will take 10-30 minutes per algorithm!\n";
                cout << "Continue? (y/n): ";
                char confirm;

                if (!get_valid_char(confirm)) {
                    cout << "\n[ERROR] Invalid input. Cancelled.\n";
                    continue;
                }

                if (confirm != 'y' && confirm != 'Y') {
                    cout << "[INFO] Cancelled.\n";
                    continue;
                }
            }

            // Generate random matrix
            vector<vector<double>> A(size, vector<double>(size));
            cout << "\n[INFO] Generating random " << size << "x" << size << " matrix...\n";
            cout << "[INFO] Matrix will include:\n";
            cout << "         - Negative values: YES\n";
            cout << "         - Floating-point values: YES\n";
            cout << "         - Null values: NO (replaced with valid numbers)\n";
            cout << "         - Diagonal dominance: YES (ensures invertibility)\n";

            generate_random_matrix(A, size);

            // Verify it's invertible
            cout << "[INFO] Verifying matrix invertibility...\n";
            double det = calculate_determinant(A, size);

            if (abs(det) < 1e-10) {
                cout << "[WARNING] Generated matrix is nearly singular. Regenerating...\n";
                generate_random_matrix(A, size);
                det = calculate_determinant(A, size);
            }

            cout << "[INFO] Matrix is invertible (det not equal to 0)\n";

            // Run the benchmark
            run_single_benchmark(size, A, num_threads);

            // Ask to continue
            cout << "\nPress Enter to return to menu...";
            cin.get();
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    cout << "\n" << string(70, '#') << "\n";
    cout << "##  MATRIX INVERSION: GAUSS-JORDAN vs LU DECOMPOSITION COMPARISON   ##\n";
    cout << "##                                                                  ##\n";
    cout << "##  Comparing Two Algorithms:                                       ##\n";
    cout << "##    1. Gauss-Jordan Method (Simple, High Parallelism)             ##\n";
    cout << "##    2. LU Decomposition (Numerically Stable, Complex)             ##\n";
    cout << "##                                                                  ##\n";
    cout << "##  Each Algorithm Tested With:                                     ##\n";
    cout << "##    - Serial (Single-threaded Baseline)                           ##\n";
    cout << "##    - OpenMP (Shared Memory Parallel)                             ##\n";
    cout << string(70, '#') << "\n";

    // Initialize thread settings (local variable, no global)
    int max_threads = get_max_threads();
    int num_threads = 0;  // 0 means use maximum threads (default)

    cout << "\n[INFO] System has " << max_threads << " thread(s) available\n";
    cout << "[INFO] Default setting: Using all " << max_threads << " threads\n";
    cout << "[INFO] You can change this in the Thread Configuration menu\n";

    while (true) {
        display_main_menu(num_threads);

        cout << "\nEnter your choice (0-3): ";
        int choice;

        if (!get_valid_int(choice, 0, 3)) {
            cout << "\n[ERROR] Invalid input! Please enter a number between 0 and 3.\n";
            cout << "Press Enter to continue...";
            cin.get();
            continue;
        }

        if (choice == 0) {
            cout << "\n[INFO] Exiting program. Thank you!\n\n";
            break;
        }
        else if (choice == 1) {
            handle_static_dataset(num_threads);

            cout << "\nPress Enter to return to main menu...";
            cin.get();
        }
        else if (choice == 2) {
            handle_dynamic_dataset(num_threads);
        }
        else if (choice == 3) {
            handle_thread_configuration(num_threads);
        }
    }

    return 0;
}
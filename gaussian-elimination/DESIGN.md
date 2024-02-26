# Parallelization Design Documentation: Gaussian Elimination 

## 1. Mathematical Background
Gaussian Elimination is a method for solving systems of linear equations of the form $A\times x=b$ where $A$ is an $n\times n$ matrix and $b$ is a vector of constants.

The unsolved equation is transformed into the form of $U\times x=y$, where $U$ is an upper triangular matrix whose diagonal elements have the value 1. $y$ is the augmented version of vector $b$ whose values are updated during the back substitution stage.

## 2. Theoretical Analysis
The provided serial code `gauss.c` contains the following function `gauss()`: 

```c
void gauss()
{
    int norm, row, col; /* Normalization row, and zeroing
                         * element row and col */
    float multiplier;

    printf("Computing Serially.\n");

    /* Gaussian elimination */
    for (norm = 0; norm < N - 1; norm++)
    {
        for (row = norm + 1; row < N; row++)
        {
            multiplier = A[row][norm] / A[norm][norm];
            for (col = norm; col < N; col++)
            {
                A[row][col] -= A[norm][col] * multiplier;
            }
            B[row] -= B[norm] * multiplier;
        }
    }
    /* (Diagonal elements are not normalized to 1.  This is treated in back
     * substitution.)
     */

    /* Back substitution */
    for (row = N - 1; row >= 0; row--)
    {
        X[row] = B[row];
        for (col = N - 1; col > row; col--)
        {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
}
```
Comprised of a gaussian elimination step involving $O(n^3)$ operations and a back substitution stage only involving $O(n^2)$ operations. This difference in time complexity allows us to focus our parallelization efforts solely on the first step.

### 2.1 Speedup and Scalability Analysis
We begin parallelization design by first performing fixed-size speedup and scalability analysis on a pseudo-equivalent gaussian elimination function. This theoretical analysis will inform early design decision for subsequent parallelization design in `pthreads` and `OpenMP`.

#### 2.1.1 Serial Analysis
For the following pseudo-code function `serial_gauss()` derived from the serial code `gauss.c`:

```c
void serial_gauss() {
  int N;
  volatile float A[N][N], B[N], X[N]; 

  int norm, row, col;
  float multiplier;

  for (norm = 0; norm < N -1; norm++) {
    for (row = norm + 1; row < N; row++) {
      multiplier = A[row][norm] / A[norm][norm];
      for (col = norm; col < N; col++) {
        A[row][col] -= A[norm][col] * multiplier;
      }
      B[row] -= B[norm] * multiplier;
    }
  }
}
```
**Operation Summary:**
- An outer loop that iterates `N` times
- A middle loop that iterates `N` times and performs two (3) mathematical operation
- An inner loop that iterates `N` times and performs two (2) mathematical operations

Therefore, the serial runtime, $T_s$, is $$T_s=2N^3+3N^2,$$ which corresponds to two operations performed $N\times N\times N$ times, and three operations performed $N\times N$ times.

#### 2.1.2 Parallel Derivation and Analysis
Prior to the derivation of parallel runtime, $T_p$, it is cognizant to obtain any *loop-carried dependence* in the `abstract_gauss()` function by unrolling the inner and middle loop.

**Middle Loop Rollout on `row` @ `norm=0`:**
```c
multiplier = A[1][0]/A[0][0];    B[1]-=B[0] * multiplier    //row = 1
multiplier = A[2][0]/A[0][0];    B[2]-=B[0] * multiplier    //row = 2
multiplier = A[3][0]/A[0][0];    B[3]-=B[0] * multiplier    //row = 3
//...
multiplier = A[N-1][0]/A[0][0];  B[3]-=B[0] * multiplier    //row = N-1
```
**Inner Loop Rollout on `col` @ `norm=0` and `row=1`:**
```c
A[1][0] -= A[0][0] * multiplier       //col = 0
A[1][1] -= A[0][1] * multiplier       //col = 1
A[1][2] -= A[0][2] * multiplier       //col = 2
//...
A[1][N-1] -= A[0][N-1] * multiplier   //col = N-1 
```
Upon inspection of both the inner and middle loop of the gaussian elimination step, it **appears that neither rollout exhibits loop-carried dependence**. However, write operations on elements of matrix `A` and vector `B` are dependent on read operations on themselves respectively. Finally, reads on `A` and `B` occur on regions of the arrays that are not written to during during loop iterations, and the inner loop is read dependent on the outer loop, owing to shared variable `multiplier`.

With these observations in mind, we can begin to derive an estimated formulation for parallel runtime $T_p$. Parallelization with $p$ processors will be on the **middle loop**, and each processor will perform computations on $N/p$ rows of matrix $A$. Therefore, the simplified parallel runtime, $T_p$, is

$$T_p=\frac{2N^3}{p}+\frac{3N^2}{p} = \frac{2N^3+3N^2}{p}.$$

#### 2.1.3 Speedup
We can now compute the theoretical speedup, $s_p$, of the parallelization:
```math
\begin{gather*}
  s_p = \frac{T_s}{T_p}\\
  = \frac{2N^3+3N^2}{\frac{2N^3+3N^2}{p}}\\
  = p
\end{gather*}
```
#### 2.1.4 Scalability Analysis
We will now determine if the parallelization as formulated is effective at scale by equating $s_p$ with its relationship to Amdahl's fraction, $\alpha$ and the number of processors, $p$.
```math
\begin{gather*}
  s_p = \frac{1}{\alpha+\frac{1-\alpha}{p}}\\
  p = \frac{1}{\alpha+\frac{1-\alpha}{p}}\\
  p\alpha + 1 - \alpha = 1\\
  \alpha(p-1) = 0\\
  \therefore \alpha = 0
\end{gather*}
```
### 2.2 Discussion
The results of the theoretical speedup and scalability analysis implies that an idealized parallelization of the gaussian elimination function would result in a *linear* speedup with perfect efficiency, such that all work can be effectively parallelized. In addition to speedup, another pertinent metric to discuss is runtime, for various processor, $p$, and workload $N$, cases. The below figure illustrates the hypothetical $T_p$ for $p\in [1,\text{ }128]$ and $N\in[100,\text{ }5{,}000]$.

<p>
    <img src="/gaussian-elimination/analysis/figures/parallel-runtime.png">
    <em>asdf</em>
</p>

Theoretically, a log-linear relationship exists between the number of processors and operation runtime exists, such that runtime proportionally decreases with the number of processors used. Whether that be the case in practice is subject to experimental consideration.

## 3. Design
The design and experimentation environment for this experiment was a virtual instance CentOS 8 hosted by [Chameleon Cloud](https://www.chameleoncloud.org/). The instance was physically hosted by a Dell PowerEdge R740 compute node on the Chameleon Cluster, with access to two (2) [Intel® Xeon® Gold 6126](https://www.intel.com/content/www/us/en/products/sku/120483/intel-xeon-gold-6126-processor-19-25m-cache-2-60-ghz/specifications.html) Processors. Intel's website note the following CPU Specifications:

| Parameter                 | Value              |
|---------------------------|----------------------|
| Total Cores               | 12                   |
| Total Threads             | 24                   |
| Max Turbo Frequency       | 3.70 GHz             |
| Processor Base Frequency  | 2.60 GHz             |
| Cache                     | 19.25 MB L3 Cache    |
| Max # of UPI Links        | 3                    |
| TDP                       | 125 W                |



### 3.1 Initial Design in `OpenMP`
Parallelization design began with implementing the source code `gauss.c` with `OpenMP`, a directive-based parallel programming model. The initial design `gauss-parallel.c` consisted of modifications to the source code in the form of compiler directives and minimal changes to the source code.

```c
// 1. Include header file
#include <omp.h>

void gauss()
{
    // 1.1 Instantiate private variables for loop indexing
    int norm, row, col; 
    float multiplier;

    // 1.2 Specify number of threads
    int threads = 2;

    printf("Computing in Parallel.\n");

    // 2. Begin parallel region for the scope of the gaussian-elimination step
    #pragma omp parallel num_threads(threads) shared(N, A, B, X, threads) private(norm, row, col, multiplier) default(none)
    {
		
	// 2.1 Log number of processors and threads used by OpenMP
	#pragma omp single nowait
	{
	int num_threads = omp_get_num_threads();
	printf("Threads: %d\n", num_threads);
	}	
		
        for (norm = 0; norm < N - 1; norm++)
        {
            // 3. Begin parallel directive
            #pragma omp for
            for (row = norm + 1; row < N; row++)
            {
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++)
                {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
            }
        }
    }
    /* Back substitution (NOT SUBJECT TO PARALLELIZATION)*/
    for (row = N - 1; row >= 0; row--)
    {
        X[row] = B[row];
        for (col = N - 1; col > row; col--)
        {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
}
```
This parallelization includes the following steps:
- **1.0:** The header line `#include <omp.h>` points the compiler towards `OpenMP`
- **1.1:** Variables `norm, row, col, multiplier` are instantiated for loop indexing and operations
- **1.2:** The number of `threads` are specified for parallelization
- **2.0:** The scope of the parallel region is defined using the directive `#pragma omp parallel num_threads(threads) shared(N, A, B, X, threads) private(norm, row, col, multiplier) default(none)`. This directive includes the prefix to denote shared variables related to the computation of the gaussian-elimination step, and private variables for individual thread processes.
- **2.1:** The `#pragma omp single nowait` directive was used for debugging purposes, to confirm that the number of specified threads are active using the program printout. The `nowait` clause allows for one thread to perform the `single` section and the rest of the threads to proceed to the rest of the parallel region.
- **3.0** The parallel directive `#pragma omp for` specifies the beginning of parallel directive at the middle loop.

### 3.2 Design Validation
The parallelized version, `gauss-parallel.c`, was compared against the original source code for computational correctness. The source code `gauss.c` was compiled using the command `gcc -o gauss.out gauss.c`. The generated executable was run with the parameters `N=10` and `seed=2` with the command `./gauss.c 10 2` to obtain initial reproducible results for comparison.

Similarly, the parallelized version was compiled with the command `gcc -o gauss-parallel.out -fopenmp gauss-parllel.c`. The flag `-fopenmp` directs the compiler to use `OpenMP` when generating the executable. The parallelization was evaluated against the serial source code with the command `./gauss-parallel.out 10 2` and both programs resulted in the same identical results for `X`:
```
Program Printout:
X = [ 0.97; -0.94;  0.93;  0.29; -1.09;  0.35; -1.45; -0.57;  1.79;  1.36]
```
This initial validation confirmed that validity of the initial `OpenMP` parallelization design, and each future iteration of parallelization was validated this way.

### 3.3 Initial Experimental Results
The parallelized `OpenMP` program was ran for a number of threads `threads` and workload size `N` for `threads` for values identical to the theoretical analysis in section **2.2**.

<p>
    <img src="/gaussian-elimination/analysis/figures/experiment1.png">
    <em>Gaussian-Elimination Runtimes for Workload-Thread Count Cases</em>
</p>

The results of this experiment differed from the theoretical analysis and two regimes emerged with respect to runtime, workload, and thread count. For workloads $N\in[500,\text{ }2{,}000]$, runtime decreased with thread count up to 16 threads, plateaued in runtime up to 32 threads, and actually **increased** in runtime up to 128 threads. This can most likely be attributed to when thread count is is over 32, thread management overhead is costly enough to detriment performance.

The second regime is for the upper range of workloads $N\in[500,\text{ }2{,}000]$, whose runtime similarly decreased up to 16 threads, then plateaued for thread counts between 16 to 128. This plateau in performance suggests that at larger workloads there is significant sequential operations that can not be minimized (possibly the back substitution step), or similar to the previous regime, thread management overhead creates significant runtime. There is also the possibility the performance behavior at higher workloads may actually follow the first regime at higher thread counts, but due to the time costs of evaluation and the scope of this experimentation this was not further explored.

The consequences of thread management overhead are especially visible in the smallest workload scenario when $N=100$, where runtime jumps an order of magnitude between 16 and 32 threads. As evident in the CPU summary, since each processor has 24 threads any addition in thread count will result accruing overhead costs. For thread counts above 32, it is most likely the case that thread management overhead creates at minimum 10 milliseconds of additional program runtime.

### 3.4 Performance Evaluation and Design Objectives
The experimental results in section 3.3 confirm two assumptions that will be used to evaluate parallelization performance and further design objectives. First is that performance generally increases with thread count up to 32 threads. Second is that after 24 threads, additional overhead requires at least 10 milliseconds of runtime. These assumptions essentially dictate an "upper and lower bound" from which we try to optimize runtime for.

With respect to speedup, these bounds are illustrates between the theoretical 1:1 speedup and experimentally recorded speedup:

<p>
    <img src="/gaussian-elimination/analysis/figures/experiment1-sp.png">
    <em>Experimentally Recorded Speedup</em>
</p>

The experimental results indicate proportional speedup for all workloads up to 16 threads, with pronounced decline in speed up with higher thread count. The ideal optimization would therefore be bounded below the theoretical linear speedup and above the recorded data.

With these considerations in mind, optimization will be evaluated for a workload of `N = 200` under the assumption that performance gains will scale accordingly. In addition, Upon final design we will revisit this assumption.

### 3.5 Design Improvements
The below design improvements have been made to the original parallelization implementation:
```c
// 1. Include header file
#include <omp.h>

void gauss()
{
    // 1.1 Instantiate private variables for loop indexing
    int norm, row, col; 
    float multiplier;

    // 1.2 Specify number of threads
    int threads = 2;

    printf("Computing in Parallel.\n");

    // 2. Begin parallel region for the scope of the gaussian-elimination step
    #pragma omp parallel num_threads(threads) shared(N, A, B, X, threads) private(norm, row, col, multiplier) default(none)
    {
		
	// 2.1 Log number of processors and threads used by OpenMP
	#pragma omp single nowait
	{
	int num_threads = omp_get_num_threads();
	printf("Threads: %d\n", num_threads);
	}	
		
        for (norm = 0; norm < N - 1; norm++)
        {
            // 3. Begin parallel directive
            #pragma omp for
            for (row = norm + 1; row < N; row++)
            {
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++)
                {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
            }
        }
    }
    /* Back substitution (NOT SUBJECT TO PARALLELIZATION)*/
    for (row = N - 1; row >= 0; row--)
    {
        X[row] = B[row];
        for (col = N - 1; col > row; col--)
        {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
}
```
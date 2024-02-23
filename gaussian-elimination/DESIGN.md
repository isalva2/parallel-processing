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
The results of the theoretical speedup and scalability analysis implies that an idealized parallelization of the gaussian elimination function would result in a *linear* speedup with perfect efficiency, such that all work can be effectively parallelized. In addition to speedup, another pertinent metric to discuss is runtime, for various processor, $p$, and workload $N$, cases. The below figure illustrates the hypothetical $T_p$ for $p\in [1,\text{ }128]$ and $N\in[100,\text{ }20{,}000]$.


## 3. Design
The design and experimentation environment for this experiment was a virtual instance CentOS 8 hosted by [Chameleon Cloud](https://www.chameleoncloud.org/). The instance was physically hosted by a Dell PowerEdge R740 compute node on the Chameleon Cluster, with access to two (2) Intel® Xeon® Gold 6126 Processors.

### 3.1 Initial Design in `OpenMP`
Parallelization design began with implementing the source code `gauss.c` with `OpenMP`, a directive-based parallel programming model. The initial design consisted of only minor modifications to the source code in the form of compiler directives.
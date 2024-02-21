# Parallelization Design Documentation: Gaussian Elimination 

## Mathematical Background
Gaussian Elimination is a method for solving systems of linear equations of the form $A\times x=b$ where $A$ is an $n\times n$ matrix and $b$ is a vector of constants.

The unsolved equation is transformed into the form of $U\times x=y$, where $U$ is an upper triangular matrix whose diagonal elements have the value 1. $y$ is the augmented version of vector $b$ whose values are updated during the back substitution stage.

## Theoretical Analysis
The provided serial code `gauss.c` contains the following function: 

```c
void gauss() {
  int norm, row, col;  /* Normalization row, and zeroing
			* element row and col */
  float multiplier;

  printf("Computing Serially.\n");

  /* Gaussian elimination */
  for (norm = 0; norm < N - 1; norm++) {
    for (row = norm + 1; row < N; row++) {
      multiplier = A[row][norm] / A[norm][norm];
      for (col = norm; col < N; col++) {
	      A[row][col] -= A[norm][col] * multiplier;
      }
      B[row] -= B[norm] * multiplier;
    }
  }
  /* (Diagonal elements are not normalized to 1.  This is treated in back
   * substitution.)
   */

  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}
```

Comprised of a gaussian elimination step involving $O(n^3)$ operations and a back substitution stage only involving $O(n^2)$ operations. This difference in time complexity allows us to focus our parallelization efforts solely on the first step.

### Fixed-Size Speedup and Scalability Analysis
We begin parallelization design by first performing fixed-size speedup and scalability analysis on a pseudo-equivalent gaussian elimination function. This theoretical analysis will inform early design decision for subsequent parallelization design in `pthreads` and `OpenMP`.

#### Serial Analysis
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

Therefore, the serial run time, $T_s$, is
$$T_s=2N^3+3N^2.$$

#### Parallel Derivation and Analysis
Prior to the derivation of any pseudo-code for parallelization, it is cognizant to obtain any *loop-carried dependence* in the `abstract_gauss()` function by unrolling the inner and middle loop.

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

With these observations in mind, we can begin to derive a naive pseudo-code function, `parallel_gauss()`, for parallelized gaussian elimination. Parallelization with $p$ processors will be on the **middle loop**, and each processor will perform computations on $N/p$ rows of matrix $A$. Due to read and write operations on `A` and `B`, temporary variables `tempA` and `tempB` will be used to collect and store the individual calculations of each processor. Here is the naive parallelization `parallel_gauss()`:

```c
void parallel_gauss() {
    int N, i, j, nprocs, myid;
    volatile float A[N][N], tempA[N][N], B[N], tempB[N], X[N];

    int norm, row, col;
    float multiplier;


    nprocs = no. of processors used;
    myid = process id;

    for (norm = 0; norm < N -1; norm++) {

      /*
      Temporary variables to store calculations derived from A and B.
      These variables reset to zero after each lock() unlock().
      */ 
      for (i = 0; i < N; i++) {
        tmpB[i] = 0;
        for (j = 0; j < N; j++) {
          tmpA[i][j] = 0;
        }
      }

      /*
      Store gaussian elimination calculations in tempA and tempB
      for each processor.
      */
      for (row = myid + 1; row < N; row += nprocs) {
        multiplier = A[row][norm] / A[norm][norm];
        for (col = norm; col < N; col++) {
          tempA[row][col] = A[norm][col] * multiplier;
        }
        tempB[row] = B[norm] * multiplier;
      }

      /*
      Iterate lock() and unlock() operations p times to update
      global A and B.
      */
      lock();
      for (i = 0; i < N; i++) {
        B[i] -= tmpB[i];
          for (j = 0; j < N; j++) {
            A[i][j] -= tempA[i][j];
          }
      }
      unlock();
    }
}
```
**Operation Summary:**
- An outer loop that iterates `N` times with negligible assignment time to `tempA` and `tempB`.
- A middle loop that iterates $N/p$ times and performs two (2) operations
- An inner loop that iterates N times and performs one (1) operation.
- A `lock()`/`unlock()` **critical region**, that iterates $p$ times and performs one (1) operation, and an inner loop in the **critical region** that performs `N` operations.

Therefore, the run time, $T_p$, for this parallelized gaussian elimination formulation is

$$T_p=\frac{N^3}{p}+\frac{2N^2}{p}+pN^2+pN = \frac{N^3+2N^2+p^2N^2+p^2N}{p}$$

#### Speedup
We can now compute the theoretical speedup, $s_p$, of the parallelization:
```math
\begin{gather*}
  s_p = \frac{T_s}{T_p}\\
  = \frac{2N^3+3N^2}{\frac{N^3+2N^2+p^2N^2+p^2N}{p}}\\
  = \frac{pN(2N^2+3N)}{N(N^2+2N+p^2N+p^2)}\\
  \therefore s_p = \frac{2pN^2+3pN}{N^2+2N+p^2N+p^2}
\end{gather*}
```
#### Scalability Analysis
We will now determine if the parallelization as formulated is effective at scale by equating $s_p$ with its relationship to Amdahl's fraction, $\alpha$ and the number of processors, $p$.
```math
\begin{gather*}
  s_p = \frac{1}{\alpha+\frac{1-\alpha}{p}}\\
  \frac{2pN^2+3pN}{N^2+2N+p^2N+p^2} = \frac{1}{\alpha+\frac{1-\alpha}{p}}\\
\end{gather*}
```
Solving symbolically for $s_p$ with [WolframAlpha](https://www.wolframalpha.com/), we see that 
$$\alpha = \frac{p+1}{2n+3},$$
and that the limit, $\lim_{N\to 0}\alpha=0$, indeed approaches zero. We have theoretically confirmed that the parallelization of the gaussian elimination step is effective.

###
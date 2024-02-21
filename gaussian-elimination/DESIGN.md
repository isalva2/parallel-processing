# Parallelization Design Documentation: Gaussian Elimination 

## Mathematical Background
Gaussian Elimination is a method for solving systems of linear equations of the form $A\times x=b$ where $A$ is an $n\times n$ matrix and $b$ is a vector of constants.

The unsolved equation is transformed into the form of $U\times x=y$, where $U$ is an upper triangular matrix whose diagonal elements have the value 1. $y$ is the augmented version of vector $b$ whose values are updated during the gaussian elimination stage.

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
For the following pseudo-code function `abstract_gauss()` derived from the serial code `gauss.c`:

```c
void abstract_gauss() {
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

We observe the following operations by inspection that contribute to the run time of the function:
- An outer loop that occurs `N` times
- A middle loop that occurs `N` times and performs two (2) mathematical operation
- An inner loop that occurs `N` times and performs two (2) mathematical operations

We therefore we have a serial run time $T_s$, of
$$T_s=2N^3+2N^2.$$

#### Parallel Analysis
Prior to the derivation of any pseudo-code for parallelization, it is cognizant to obtain any *loop-carried dependence* in the `abstract_gauss()` function by unrolling the inner and middle loop.

**Middle Loop Rollout**
- `multiplier = A[1][0]/A[0][0];    B[1]-=B[0] * multiplier`
- `multiplier = A[2][1]/A[1][1];    B[2]-=B[0] * multiplier`
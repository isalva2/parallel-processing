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

Therefore, the serial run time, $T_s$, is
$$T_s=2N^3+2N^2.$$

#### Parallel Analysis
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

Upon inspection of both the inner and middle loop of the gaussian elimination step, it **appears that neither rollout exhibits loop-carried dependence**. However, write operations on elements of matrix `A` and vector `B` are dependent on read operations on themselves respectively. Finally, reads on `A` and `B` occur on regions of the arrays that are not written to during during loop iterations, and the inner loop is read dependent on the outer loop, owed shared variable `multiplier`.

With these observations, 
# README

This deliverable contains the following items:
1. Three parallelized MPI models
- Single Program Multiple Data (SPMD) using point-to-point communication `model1_pp.c`
- SPMD using collective calls `model2_cc.c`
- Task and Data Parallel Model `model3_td.c`
2. A serial version of the program `serial_convolution.c`
3. A root mean square error checking program `rsme_check_convolution.c`
4. Additional folder with report figures
5. Experimental results for Models 1 and 2

In order to run these files, the input data must be in a directory `data/` that is in the same directory as the files, and an additional directory `data/results/` to store the results of each model.

Each MPI models can be compiled using the MPI compiler `mpicc`

```bash
$ mpicc model[1, 2, 3]_[pp, cc, td].c
```
However, some systems will not be able to find the `<math.h>` file, and must be explicitly linked using the `-lm` flag at the end of the above command.

The resulting executable can be run by using `mpirun` and specifying the number of desired processes using the `-np` flag.

```bash
$ mpirun -np <processes> ./a.out
```

For convenience, the included shell file `run.sh` compiles each model, runs each model on images `2_im1` and `2_im2`, and checks the RMSE of each model. The shell file can be ran by first changing permissions on the file then executing it.

```bash
$ chmod +x run.sh
$ ./run.sh
```

The shell file runs Models 1 and 2 with four processes and Model 3 with the requisite 8. Upon completion of each model the results are stored in `data/results/`.
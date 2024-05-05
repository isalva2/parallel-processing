#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 512

void read_file(char path[], double matrix[N * N])
{
    FILE *fp = fopen(path, "r");
    for (int i  = 0; i < N * N; i++)
    {
        fscanf(fp, "%lg", &matrix[i]);
    }
    fclose(fp);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s \"model[1, 2, or 3]_out2\"\n", argv[0]);
        return 1;
    }


    const char *path = "data/results/";
    char *file_name = argv[1];
    char file_path[100];
    strcpy(file_path, path);
    strcat(file_path, file_name);

    double yhat[N * N];
    read_file(file_path, yhat);
    double y[N * N];
    read_file("data/out_2", y);

    printf("Root Mean Square Error of \"%s\"\n\n", file_name);
    printf("First 5 values of experimental (%s) and actual \"out_2\":\n", file_name);
    printf("Experimental:\t");
    for (int i = 0; i < 5; i++)
        printf("%e\t", yhat[i]);
    printf("\nActual:\t\t");
    
    for (int i = 0; i < 5; i++)
        printf("%e\t", y[i]);
    printf("\n\n");

    // RMSE check
    double square_error = 0.0;
    double RMSE;

    for (int i = 1; i < N * N; i++)
    {
        square_error += pow(yhat[i] - y[i], 2);
    }
    RMSE = sqrt(square_error / (N * N));

    printf("RMSE: %f\n", RMSE);

    return 0;
}
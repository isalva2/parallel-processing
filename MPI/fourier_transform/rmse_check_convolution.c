#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 512

void read_file(char path[], float matrix[N * N])
{
    FILE *fp = fopen(path, "r");
    for (int i  = 0; i < N * N; i++)
    {
        fscanf(fp, "%f", &matrix[i]);
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

    printf("%s", file_path);

    float y_hat[N * N];
    read_file(file_path, y_hat);
    float y[N * N];
    read_file("data/out_2", y);

    printf("\n");
    for (int i = 0; i < 10; i++)
        printf("%6.2f\t", y_hat[i]);
    printf("\n\n");
    for (int i = 0; i < 10; i++)
        printf("%6.2f\t", y[i]);
    printf("\n\n");






    return 0;
}
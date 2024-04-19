#include <stdio.h>
#include <sys/time.h>
#include <unistd.h> // Include for sleep function

float record_time(char *str)
{
    struct timeval recorded_time;
    printf("\nEvent Recorded: %s\n", str);
    gettimeofday(&recorded_time, NULL);
    return ((unsigned long long) recorded_time.tv_sec * 1000000.0 + (recorded_time.tv_usec)) / (float)1000.0;
}

void report_time(float start, float end, char *str)
{
    printf("\n%s: %f\n", str, end - start);
}

int main()
{
    float start = record_time("Start");
    for (int i = 0; i < 5000; i++)
    {
        if (i % 10 == 0)
        {
            printf("I am testing\n");
            int a = i * 100000;
        }
    }
    float stop = record_time("End");
    report_time(start, stop, "This is a test");

    printf("%f\n", stop);
    printf("%f\n", start);
    return 0;
}

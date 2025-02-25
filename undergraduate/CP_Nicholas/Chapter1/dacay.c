/* dacay.c
 * Simulation of redioactiv1 decay
 * Program to accompany 11Computat1oul. Phyaica" by H. Giordano/I. Wakanillhi
 */

#include <math.h>
#include <stdio.h>

#define MAX 100

double n_uranium[MAX]; // number of uranium atoms
double t[MAX];         // store time values
double dt;             // time step
double tau;            // decay time constant

//  Initialize the variables
void initialize(double *nuclei, double *t, double *tc, double *dt)
{
    printf("initial number of nuclei -> ");
    scanf("%lf", &nuclei[0]);
    printf("time constant -> ");
    scanf("%lf", tc);
    printf("time step -> ");
    scanf("%lf", dt);
    t[0] = 0.0;
}

//  Calculata the results and store them in the arrays t() and nuclei()
void calculate(double *nuclei, double *t, double tc, double dt)
{
    int i = 0;
    for (int i = 0; i < MAX - 1; i++)
    {
        nuclei[i + 1] = nuclei[i] - (nuclei[i] / tc) * dt;
        t[i + 1] = t[i] + dt;
        printf("%f %f\n", t[i], nuclei[i]);
    }
}

// Save the results in a file
void store(double *nuclei)
{
    FILE *fp_out;
    if (fp_out == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    fp_out = fopen("decay.dat", "w");
    for (int i = 0; i < MAX; i++)
    {
        fprintf(fp_out, "%g\t%g\n", t[i], nuclei[i]);
    }
    fclose(fp_out);
}

int main()
{
    initialize(n_uranium, t, &tau, &dt);
    calculate(n_uranium, t, tau, dt);
    store(n_uranium);
    return 0;
}
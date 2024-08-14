#include "params.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>    /* For OpenMP & MPI implementations, use their walltime! */
#include "mpi.h"

// Define a global variable to store the size of each process
int N1_local;

/*
void init(double u[N1][N2][N3]) {
    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                u[n1][n2][n3] = u0(n1, n2, n3);
            }
        }
    }
};

void dudt(const double u[N1][N2][N3], double du[N1][N2][N3]) {
    double sum;
    int count;
    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                sum = 0.0;
                count = 0;
                for (int l1 = imax(0, n1 - ml); l1 <= imin(n1 + ml, N1 - 1); l1++) {
                    for (int l2 = imax(0, n2 - ml); l2 <= imin(n2 + ml, N2 - 1); l2++) {
                        for (int l3 = imax(0, n3 - ml); l3 <= imin(n3 + ml, N3 - 1); l3++) {
                            sum += u[l1][l2][l3];         // Accumulate the local sum in sum
                            count++;                      // Increment the count
                        }
                    }
                }
                du[n1][n2][n3] = (sum / count);     // store the local mean in du
            }
        }
    }
};
*/

// Rewrite the init function to make each process only initialise its own local_u, and then reduce them using MPI_Allgather
void init_local_u(double local_u[N1_local][N2][N3], int rank) {
    int start = rank * N1_local;
    for (int n1 = 0; n1 < N1_local; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                local_u[n1][n2][n3] = u0(start + n1, n2, n3);
            }
        }
    }
};

// Rewrite the dudt function to compute each process's own local_du, and then reduce them using MPI_Allgather
void dudt_local(const double u[N1][N2][N3], double local_du[N1_local][N2][N3], int rank) {
    double sum;
    int count;
    int start = rank * N1_local;
    for (int n1 = start; n1 < start + N1_local; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                sum = 0.0;
                count = 0;
                for (int l1 = imax(0, n1 - ml); l1 <= imin(n1 + ml, N1 - 1); l1++) {
                    for (int l2 = imax(0, n2 - ml); l2 <= imin(n2 + ml, N2 - 1); l2++) {
                        for (int l3 = imax(0, n3 - ml); l3 <= imin(n3 + ml, N3 - 1); l3++) {
                            sum += u[l1][l2][l3];
                            count++;
                        }
                    }
                }
                local_du[n1 - start][n2][n3] = (sum / count);     // Store the result into local_du
            }
        }
    }
}
/*
void step(double u[N1][N2][N3], const double du[N1][N2][N3]) {
    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                u[n1][n2][n3] = r * du[n1][n2][n3] * (1.0 - du[n1][n2][n3]);
            }
        }
    }
};

void stat(double *stats, const double u[N1][N2][N3]) {
    double mean = 0.0;
    double uvar = 0.0;
    double umin = 100.0;
    double umax = -100.0;
    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                mean += u[n1][n2][n3] / (N1 * N2 * N3);
                if (u[n1][n2][n3] > umax) {
                    umax = u[n1][n2][n3];
                }
                if (u[n1][n2][n3] < umin) {
                    umin = u[n1][n2][n3];
                }
            }
        }
    }
    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                uvar += (u[n1][n2][n3] - mean) * (u[n1][n2][n3] - mean) / (N1 * N2 * N3);
            }
        }
    }
    stats[0] = mean;
    stats[1] = umin;
    stats[2] = umax;
    stats[3] = uvar;
};

void write(const double u[N1][N2][N3], const int m) {
    char outfile[80];
    int fileSuccess = sprintf(outfile, "state_%i.txt", m);
    if (fileSuccess > 0) {
        FILE *fptr = fopen(outfile, "w");
        for (int n3 = 0; n3 < N3; n3++) {
            for (int n2 = 0; n2 < N2; n2++) {
                for (int n1 = 0; n1 < N1; n1++) {
                    // this segfaults when fptr is null.
                    fprintf(fptr, "%2.4f\t", u[n1][n2][n3]);
                }
                fprintf(fptr, "\n");
            }
            fprintf(fptr, "\n");
        }
    } else {
        printf("Failed to write state_%i.txt!\n", m);
    }
};
*/

int main(int argc, char **argv) {

    // Initialize the MPI program
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double u[N1][N2][N3];
    double du[N1][N2][N3];
    double stats[M / mm][4];
    int writeInd = 0;

    printf("Process %d has the u\n", rank);

    // Assume there are p processes, then split the 3 dimensions space evenly into p N1/p * N2 * N3 square block
    N1_local = N1 / size;

    // Define the local array u and du on each process
    double local_u[N1_local][N2][N3];
    double local_du[N1_local][N2][N3];

    printf("Process %d has the du\n", rank);

    if (rank == 0) {
        // Use MPI_Scatter to distribute the global array u to each process
        MPI_Scatter(u, N1_local * N2 * N3, MPI_DOUBLE,
                    local_u, N1_local * N2 * N3, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

    } else {
        MPI_Scatter(NULL, N1_local * N2 * N3, MPI_DOUBLE,
                    local_u, N1_local * N2 * N3, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    printf("Process %d has received the u_local\n", rank);

    // Each process initialises its own part of the array
    init_local_u(local_u, rank);
    printf("Initialisation finished on process %d\n", rank);

    for (int i = 0; i < N1_local; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                printf("local_u[%d][%d][%d] on process %d is %f\n", i, j, k, rank, local_u[i][j][k]);
            }
        }
    }

    // Use MPI_Gather to gather the initialised parts back to all processes
    MPI_Allgather(local_u, N1_local * N2 * N3, MPI_DOUBLE,
                  u, N1_local * N2 * N3, MPI_DOUBLE,
                  MPI_COMM_WORLD);

    printf("u is gathered into the process %d\n", rank);

    MPI_Finalize();

    for (int i = 0; i < N1_local; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                printf("u[%d][%d][%d] is %f\n", i, j, k, u[i][j][k]);
            }
        }
    }
    clock_t t0 = clock();                   // for timing serial code
    /*
    for (int m = 0; m < M; m++) {
        // Use MPI_Scatter to distribute the global array u to each process
        // MPI_Scatter(du, N1_local * N2 * N3, MPI_DOUBLE,
        //            local_du, N1_local * N2 * N3, MPI_DOUBLE,
        //            0, MPI_COMM_WORLD);
        dudt(u, du);
        step(u, du);
        if (m % mm == 0) {
            writeInd = m / mm;
            stat(&stats[writeInd][0], u);     // Compute statistics and store in stat
        }


        write(u, m);                        // Slow diagnostic output!

    }
    double t1 = (double) (clock() - t0) / (CLOCKS_PER_SEC);     // for timing serial code

    FILE *fptr = fopen("part2.dat", "w");
    fprintf(fptr, "iter\t\tmean\t\tmin\t\tmax\t\tvar\n");  // write stats to file
    for (int m = 0; m < (M / mm); m++) {
        fprintf(fptr, "%6.0f\t%02.5f\t%02.5f\t%02.5f\t%02.5f\n",
                (double) (m * mm), stats[m][0], stats[m][1], stats[m][2], stats[m][3]);
    }
    fclose(fptr);

    double t2 = (double) (clock() - t0) / (CLOCKS_PER_SEC) - t1; // timing writes
    printf("(%3d,%3d,%3d): average iteration time per element:\t%02.16fs\n",
           N1, N2, N3, t1 / (N1 * N2 * N3 * M));
    printf("(%5d,%3d,%1d): average write time per element:\t\t%02.16fs\n",
           M, mm, 4, t2 / (4 * M / mm));

    //MPI_Finalize();

    */
    return 0;
};

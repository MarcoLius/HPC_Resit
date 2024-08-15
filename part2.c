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
};

void step_local(double local_u[N1_local][N2][N3], const double local_du[N1_local][N2][N3]) {
    for (int n1 = 0; n1 < N1_local; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                local_u[n1][n2][n3] = r * local_du[n1][n2][n3] * (1.0 - local_du[n1][n2][n3]);
            }
        }
    }
};

void stat_local(double *stats, const double local_u[N1_local][N2][N3]) {
    double mean = 0.0;
    double uvar = 0.0;
    double umin = 100.0;
    double umax = -100.0;
    for (int n1 = 0; n1 < N1_local; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                mean += local_u[n1][n2][n3] / (N1 * N2 * N3);
                if (local_u[n1][n2][n3] > umax) {
                    umax = local_u[n1][n2][n3];
                }
                if (local_u[n1][n2][n3] < umin) {
                    umin = local_u[n1][n2][n3];
                }
            }
        }
    }
    // Use MPI_Allreduce function to reduce each part of statistical data into one value
    MPI_Allreduce(&mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&umax, &umax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&umin, &umin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    for (int n1 = 0; n1 < N1_local; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            for (int n3 = 0; n3 < N3; n3++) {
                uvar += (local_u[n1][n2][n3] - mean) * (local_u[n1][n2][n3] - mean) / (N1 * N2 * N3);
            }
        }
    }

    MPI_Allreduce(&uvar, &uvar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    stats[0] = mean;
    stats[1] = umin;
    stats[2] = umax;
    stats[3] = uvar;
};


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

    // Assume there are p processes, then split the 3 dimensions space evenly into p N1/p * N2 * N3 square block
    N1_local = N1 / size;

    // Define the local array u and du on each process
    double local_u[N1_local][N2][N3];
    double local_du[N1_local][N2][N3];

    // Use MPI_Scatter to distribute the array u to each process
    MPI_Scatter(u, N1_local * N2 * N3, MPI_DOUBLE,
                local_u, N1_local * N2 * N3, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Each process initialises its own part of the array
    init_local_u(local_u, rank);

    // Use MPI_Gather to gather the initialised parts back to all processes
    MPI_Allgather(local_u, N1_local * N2 * N3, MPI_DOUBLE,
                  u, N1_local * N2 * N3, MPI_DOUBLE,
                  MPI_COMM_WORLD);

    clock_t t0 = clock();                   // for timing serial code

    for (int m = 0; m < M; m++) {
        // Use the complete array u on each process to compute the local_du, for avoiding the boundary problem
        dudt_local(u, local_du, rank);

        // Use MPI_Scatter to distribute the array u to each process for computations below
        MPI_Scatter(u, N1_local * N2 * N3, MPI_DOUBLE,
                    local_u, N1_local * N2 * N3, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Update the local_u on each process by each own local_du
        step_local(local_u, local_du);

        if (m % mm == 0) {
            writeInd = m / mm;
            stat_local(&stats[writeInd][0], local_u);     // Compute statistics on each process, reduce them and store in stat
        }

        // Use MPI_Gather to gather the local_u back to all processes
        MPI_Allgather(local_u, N1_local * N2 * N3, MPI_DOUBLE,
                      u, N1_local * N2 * N3, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        //write(u, m);                        // Slow diagnostic output!

    }
    double t1 = (double) (clock() - t0) / (CLOCKS_PER_SEC);     // for timing serial code

    if (rank == 0) {

        FILE *fptr = fopen("part2.dat", "w");
        fprintf(fptr, "iter\t\tmean\t\tmin\t\tmax\t\tvar\n");  // write stats to file
        for (int m = 0; m < (M / mm); m++) {
            fprintf(fptr, "%6.0f\t%02.5f\t%02.5f\t%02.5f\t%02.5f\n",
                    (double) (m * mm), stats[m][0], stats[m][1], stats[m][2], stats[m][3]);
        }
        fclose(fptr);

        double t2 = (double) (clock() - t0) / (CLOCKS_PER_SEC) - t1; // timing writes
        printf("(%3d,%3d,%3d): average iteration time per element:\t%02.16fs\n",
               N1, N2, N3, t1 / (long long)(N1 * N2 * N3 * M));
        printf("(%5d,%3d,%1d): average write time per element:\t\t%02.16fs\n",
               M, mm, 4, t2 / (4 * M / mm));
    }

    // Deprecate using MPI_IO to write the stats to file because it will randomize the order of writing
    /*
    MPI_File fptr;
    MPI_File_open(MPI_COMM_WORLD, "part2.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fptr);
    char write_buf[100];
    int write_start = rank * (M / mm / size);
    int write_end = write_start + (M / mm / size);
    int offset = 0;
    if (rank == 0) {
        sprintf(write_buf, "iter\t\tmean\t\tmin\t\tmax\t\tvar\n");
        MPI_File_write_at(fptr, offset, write_buf, 100, MPI_CHAR, MPI_STATUS_IGNORE);
        offset += 100;
    }
    for (int m = write_start; m < write_end; m++) {
        sprintf(write_buf, "%6.0f\t%02.5f\t%02.5f\t%02.5f\t%02.5f\n",
                (double) (m * mm), stats[m][0], stats[m][1], stats[m][2], stats[m][3]);
        MPI_File_write_at(fptr, offset, write_buf, 100, MPI_CHAR, MPI_STATUS_IGNORE);
        offset += 100;
    }
    MPI_File_close(&fptr);
    if (rank == 0) {
        double t2 = (double) (clock() - t0) / (CLOCKS_PER_SEC) - t1; // timing writes
        printf("(%3d,%3d,%3d): average iteration time per element:\t%02.16fs\n",
               N1, N2, N3, t1 / (N1 * N2 * N3 * M));
        printf("(%5d,%3d,%1d): average write time per element:\t\t%02.16fs\n",
               M, mm, 4, t2 / (4 * M / mm));
    }
    */

    MPI_Finalize();
    return 0;
};

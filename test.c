#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int N1_local;

int main(int argc, char **argv) {
    double u[8][8][8];
    double du[8][8][8];
    double stats[50000][4];
    int writeInd = 0;

    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Process %d has the u\n", rank);

    N1_local = 8 / size;
    double local_u[N1_local][8][8];
    MPI_Scatter(u, N1_local * 8 * 8, MPI_DOUBLE,
                local_u, N1_local * 8 * 8, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    printf("Process %d has received the u_local\n", rank);

    for (int i = 0; i < N1_local; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                local_u[i][j][k] = 100 * i + 10 * j + k;
            }
        }
    }

    MPI_Allgather(local_u, N1_local * 8 * 8, MPI_DOUBLE,
                  u, N1_local * 8 * 8, MPI_DOUBLE,
                  MPI_COMM_WORLD);

    printf("u is gathered into the process %d\n", rank);

    return 0;
}
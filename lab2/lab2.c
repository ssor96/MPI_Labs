#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define IDX(i, j) ((i) * m + j)

int min(int a, int b) {
    return a < b? a: b;
}

int main(int argc, char **argv) {
    const double MAX_ERR = 0.0001;
    MPI_Init(&argc, &argv);
    int total, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n, m;
    FILE *fs;
    // fs = fopen("input.bin", "rb");
    // fread(&n, sizeof(int), 1, fs);
    // fread(&m, sizeof(int), 1, fs);
    n = 4000;
    m = 2000;

    int rows = (n - 2) / total + (rank < (n - 2) % total);
    int prevRowsCount = (n - 2) / total * rank + min((n - 2) % total, rank) + 1;

    double *data = (double*)malloc((rows + 2) * m * sizeof(double));

    if (data == NULL) {
        printf("%d cannot allocate memory for grid\n", rank);
        return 1;
    }

    int *sendSizes, *shiftForSend, *recvSizes, *shiftForRecv;
    sendSizes = (int *)malloc(total * sizeof(int));
    shiftForSend = (int *)malloc(total * sizeof(int));
    recvSizes = (int *)malloc(total * sizeof(int));
    shiftForRecv = (int *)malloc(total * sizeof(int));

    memset(sendSizes, 0, total * sizeof(int));
    memset(shiftForSend, 0, total * sizeof(int));
    memset(recvSizes, 0, total * sizeof(int));
    memset(shiftForRecv, 0, total * sizeof(int));

    if (rank > 0) {
        sendSizes[rank - 1] = m;
        shiftForSend[rank - 1] = IDX(1, 0);
        recvSizes[rank - 1] = m;
        shiftForRecv[rank - 1] = IDX(0, 0);
    }
    if (rank < total - 1) {
        sendSizes[rank + 1] = m;
        shiftForSend[rank + 1] = IDX(rows, 0);
        recvSizes[rank + 1] = m;
        shiftForRecv[rank + 1] = IDX(rows + 1, 0);
    }

    // fseek(fs, (prevRowsCount - 1) * m * sizeof(double), SEEK_CUR);
    // fread(data, sizeof(double), (rows + 2) * m, fs);
    // fclose(fs);

    memset(data, 0, (rows + 2) * m * sizeof(double));

    for (int i = 0; i < rows + 2; ++i) {
        data[IDX(i, 0)] = data[IDX(i, m - 1)] = 1;
        if (prevRowsCount - 1 + i == 0 || prevRowsCount - 1 + i == n - 1) {
            for (int j = 1; j < m - 1; ++j) {
                data[IDX(i, j)] = 1;
            }
        }
    }
    MPI_Status status;
    // printf("I'm %d of %d. My rows from %d to %d. Read from %d to %d\n", rank, total, prevRowsCount, prevRowsCount + rows - 1, prevRowsCount - 1, prevRowsCount + rows);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    double err = MAX_ERR + 1;
    int iter;
    for (iter = 1; err > MAX_ERR; ++iter) {
        double localErr = 0;
        for (int step = 0; step < 2; ++step) {
            for (int i = 1; i <= rows; ++i) {
                for (int j = 1 + (step ^ ((prevRowsCount - 1 + i) & 1)); j < m - 1; j += 2) {
                    double newVal = (data[IDX(i - 1, j)] + data[IDX(i + 1, j)] 
                                     + data[IDX(i, j - 1)] + data[IDX(i, j + 1)]) / 4;
                    localErr = fmax(localErr, fabs(data[IDX(i, j)] - newVal));
                    data[IDX(i, j)] = newVal;
                }
            }
            MPI_Alltoallv(data, sendSizes, shiftForSend, MPI_DOUBLE, data, recvSizes, shiftForRecv, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        MPI_Allreduce(&localErr, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%lf %d\n", MPI_Wtime() - start, iter);
        fs = fopen("output.bin", "w");
        fclose(fs);
    }

    for (int i = 0; i < total; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == i) {
            fs = fopen("output.bin", "ab");
            if (rank == 0) {
                fwrite(data, sizeof(double), m, fs);
            }
            fwrite(data + IDX(1, 0), sizeof(double), rows * m, fs);
            if (rank == total - 1) {
                fwrite(data + IDX(rows + 1, 0), sizeof(double), m, fs);
            }
            fclose(fs);
        }
    }
    MPI_Finalize();
    free(data);
    free(sendSizes);
    free(shiftForSend);
    free(recvSizes);
    free(shiftForRecv);
    return 0;
}

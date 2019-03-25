#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define IDX(i, j) ((i) * m + j)

int min(int a, int b) {
    return a < b? a: b;
}

int main(int argc, char **argv) {
    const int MAX_ITER = 2000;
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

    int bufSz = 2 * (m * sizeof(double) + MPI_BSEND_OVERHEAD);
    void *buf = malloc(bufSz);

    if (buf == NULL) {
        printf("%d cannot allocate buf\n", rank);
        return 1;
    }

    MPI_Buffer_attach(buf, bufSz);

    int rows = (n - 2) / total + (rank < (n - 2) % total);
    int prevRowsCount = (n - 2) / total * rank + min((n - 2) % total, rank) + 1;

    double *data = (double*)malloc((rows + 2) * m * sizeof(double));

    if (data == NULL) {
        printf("%d cannot allocate memory for grid\n", rank);
        return 1;
    }

    // fseek(fs, (prevRowsCount - 1) * m * sizeof(double), SEEK_CUR);
    // fread(data, sizeof(double), (rows + 2) * m, fs);
    // fclose(fs);

    memset(data, 0, (rows + 2) * m * sizeof(double));

    for (int i = 0; i < rows + 2; ++i) {
        data[IDX(i, 0)] = data[IDX(i, m - 1)] = 1;
        if (prevRowsCount - 1 + i == 0 || prevRowsCount - 1 + i == n - 1) {
            for (int j = 1; j < m - 1; ++j) {
                data[IDX(i, j)] = prevRowsCount + i;
            }
        }
    }
    MPI_Status status;
    // printf("I'm %d of %d. My rows from %d to %d. Read from %d to %d\n", rank, total, prevRowsCount, prevRowsCount + rows - 1, prevRowsCount - 1, prevRowsCount + rows);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int iter = 1; iter <= MAX_ITER; ++iter) {
        for (int i = 1; i <= rows; ++i) {
            for (int j = 1 + ((iter & 1) ^ ((prevRowsCount - 1 + i) & 1)); j < m - 1; j += 2) {
                data[IDX(i, j)] = (data[IDX(i - 1, j)] + data[IDX(i + 1, j)] 
                                  + data[IDX(i, j - 1)] + data[IDX(i, j + 1)]) / 4;
            }
        }
        MPI_Buffer_detach(&buf, &bufSz);
        MPI_Buffer_attach(buf, bufSz);
        if (rank > 0) {
            MPI_Bsend(data + IDX(1, 0), m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        }
        if (rank < total - 1) {
            MPI_Bsend(data + IDX(rows, 0), m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        }
        if (rank > 0) {
            MPI_Recv(data, m, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        if (rank < total - 1) {
            MPI_Recv(data + IDX(rows + 1, 0), m, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%lf\n", MPI_Wtime() - start);
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
    free(data);
    MPI_Buffer_detach(&buf, &bufSz);
    free(buf);
    MPI_Finalize();
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct TSparseMatrix {
    int k;
    int *rowIdx, *columnIdx;
    double *value;
};

typedef struct TSparseMatrix SparseMatrix;

void readSparseMatrix(int k, FILE *file, SparseMatrix *matrix) {
    matrix->rowIdx = (int*)malloc(k * sizeof(int));
    matrix->columnIdx = (int*)malloc(k * sizeof(int));
    matrix->value = (double*)malloc(k * sizeof(double));
    for (int i = 0; i < k; ++i) {
        fread(matrix->rowIdx + i, sizeof(int), 1, file);
        fread(matrix->columnIdx + i, sizeof(int), 1, file);
        fread(matrix->value + i, sizeof(double), 1, file);
    }
    matrix->k = k;
}

void freeSparseMatrix(SparseMatrix *matrix) {
    free(matrix->rowIdx);
    free(matrix->columnIdx);
    free(matrix->value);
}

void multSparseMatrixByVector(SparseMatrix *matrix, double *v, double *res, int firstRowIdx) {
    for (int i = 0; i < matrix->k; ++i) {
        res[matrix->rowIdx[i] - firstRowIdx] += matrix->value[i] * v[matrix->columnIdx[i]];
//        printf("%d: %lf * %lf = %lf\n", matrix->rowIdx[i], matrix->value[i], v[matrix->columnIdx[i]], matrix->value[i] * v[matrix->columnIdx[i]]);
    }
}

void prepareMatrix(SparseMatrix *matrix, double* vec, int n) {
    double *_diag = (double *)malloc(n * sizeof(double));
    double *diag = (double *)malloc(n * sizeof(double));
    memset(_diag, 0, n * sizeof(double));
    for (int i = 0; i < matrix->k; ++i) {
        if (matrix->rowIdx[i] == matrix->columnIdx[i]) _diag[matrix->rowIdx[i]] = matrix->value[i];
    }
    memcpy(diag, _diag, n * sizeof(double));
    for (int i = 0; i < matrix->k; ++i) {
        matrix->value[i] /= -diag[matrix->rowIdx[i]];
        if (matrix->rowIdx[i] == matrix->columnIdx[i]) matrix->value[i] = 0;
    }
    for (int i = 0; i < n; ++i) {
        vec[i] /= diag[i];
    }
    free(diag);
    free(_diag);
}

int min(int a, int b) {
    return a < b? a: b;
}

int main(int argc, char **argv) {
    int total = 1, rank = 0;

    FILE *file = fopen("test.bin", "rb");

    int n;
    fread(&n, sizeof(int), 1, file);

    int k;
    fread(&k, sizeof(int), 1, file);

    int myElements = k / total + (k % total < rank);
    int firstElementIndex = k / total * rank + min(rank, k % total);
    fseek(file, firstElementIndex * (2 * sizeof(int) + sizeof(double)), SEEK_CUR);

    SparseMatrix matr;
    readSparseMatrix(myElements, file, &matr);

    int myLastRow = matr.rowIdx[matr.k - 1];
    if (rank == total - 1) myLastRow = n - 1;
    int *lastRows = (int *)malloc((total + 1) * sizeof(int));
    lastRows[0] = 0;
    lastRows[rank + 1] = myLastRow;

    int *partSizes = (int *)malloc(total * sizeof(int));
    for (int i = 0; i < total; ++i) {
        partSizes[i] = lastRows[i + 1] - lastRows[i];
    }

    fseek(file, 2 * sizeof(int) + k * (2 * sizeof(int) + sizeof(double)), SEEK_SET); // to vector pos
    int myFirstRow = min(matr.rowIdx[0], lastRows[rank] + 1);
    if (rank > 0) {
        fseek(file, myFirstRow * sizeof(double), SEEK_CUR);
    } else {
        myFirstRow = 0;
    }

    double *vec = (double *)malloc((partSizes[rank] + 1) * sizeof(double));
    fread(vec, sizeof(double), partSizes[rank] + 1, file);
    fclose(file);

    prepareMatrix(&matr, vec, n);

//     x = vec + matr * x

    double *x[2];
    for (int i = 0; i < 2; ++i) {
        x[i] = (double *)malloc(n * sizeof(double));
        memset(x[i], 0, n * sizeof(double));
    }
    int cur = 0;
//    fseek(file, 2 * sizeof(int) + k * (2 * sizeof(int) + sizeof(double)), SEEK_SET); // to vector pos
//    fread(x[cur], sizeof(double), n, file);

    int needToAddLastRow = rank == 0 || partSizes[rank]; // bool

    const int MAX_ITER = 200;
    double *localPart = (double *)malloc((partSizes[rank] + 1) * sizeof(double));
    double *addFromLastRow = (double *)malloc(total * sizeof(double));
    int *shiftForDispls = (int *)malloc(total * sizeof(int));
    shiftForDispls[0] = 0;
    for (int i = 1; i < total; ++i) {
        shiftForDispls[i] = shiftForDispls[i - 1] + partSizes[i - 1];
    }

    const double MAX_ERR = 0.00001;
    double curErr = MAX_ERR + 1;
    int iter;
    for (iter = 0; curErr > MAX_ERR; ++iter) {
        memcpy(localPart, vec, partSizes[rank] * sizeof(double));
        if (needToAddLastRow) {
            localPart[partSizes[rank]] = vec[partSizes[rank]];
        } else {
            localPart[partSizes[rank]] = 0;
        }
        multSparseMatrixByVector(&matr, x[cur], localPart, myFirstRow);

        memcpy(x[cur ^ 1], localPart, partSizes[rank] * sizeof(double));
        addFromLastRow[rank] = localPart[partSizes[rank]];
        for (int i = 0; i < total; ++i) {
            x[cur ^ 1][lastRows[rank + 1]] += addFromLastRow[i];
        }
        curErr = 0;
        for (int i = 0; i < n; ++i) {
            curErr = fmax(curErr, fabs(x[cur][i] - x[cur ^ 1][i]));
        }
        memset(x[cur], 0, n * sizeof(double));
        cur ^= 1;
        printf("%d %lf\n", iter, curErr);
    }

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            printf("%lf ", x[cur][i]);
        }
        printf("\n");
        printf("%d iterations, %lf\n", iter, curErr);
    }

    for (int i = 0; i < 2; ++i) {
        free(x[i]);
    }
    free(vec);
    free(lastRows);
    free(partSizes);
    free(localPart);
    free(addFromLastRow);
    free(shiftForDispls);
    freeSparseMatrix(&matr);

    return 0;
}

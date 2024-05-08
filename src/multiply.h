#ifndef _MULTIPLY_
#define _MULTIPLY_

#include "main.h"

void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]);
typedef struct
{
    double (*matrix1)[M];
    double (*matrix2)[P];
    double (*result_matrix)[P];
    int start_row;
    int end_row;
    int end_col;
    int mid;
} ThreadData;
#endif /*_MULTIPLY_*/

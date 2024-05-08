#include "multiply.h"
#include <emmintrin.h> // SSE3

#include <thread>

#define UNROLL 4
#define BLOCKSIZEK 256
// 定义宏，根据N的值设置BLOCKSIZE和NUM_THREADS
#if N == 1024
#define BLOCKSIZE 32
#define NUM_THREADS 32
#elif N == 2048
#define BLOCKSIZE 32
#define NUM_THREADS 64
#elif N == 512
#define BLOCKSIZE 32
#define NUM_THREADS 16
#elif N == 2560
#define BLOCKSIZE 40
#define NUM_THREADS 64
#elif N == 3072
#define BLOCKSIZE 32
#define NUM_THREADS 96
#else
#define BLOCKSIZE 32
#define NUM_THREADS 1
#endif
double PackA_g[NUM_THREADS][BLOCKSIZE * BLOCKSIZEK];
double PackB_g[NUM_THREADS][BLOCKSIZEK * BLOCKSIZE];
typedef struct thread_params
{
    int start_row;
    int end_row;
    int n;
    int p;
    double *matrix1;
    double *matrix2;
    double *result_matrix;
    double *PackA;
    double *PackB;
} ThreadParams;

// ijk
void ijk_maxtrix_multiplication(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < P; j++)
        {
            double cij = result_matrix[i * n + j];
            for (k = 0; k < M; k++) // 一行乘一列
            {
                cij += matrix1[i * n + k] * matrix2[k * n + j];
            }
            result_matrix[i * n + j] = cij;
        }
    }
}
void ijk_1_4(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < P; j += 4)
        {
            // 这里实现一行乘四列
            for (k = 0; k < M; k++)
            {
                result_matrix[i * p + j] += matrix1[i * n + k] * matrix2[k * p + j];
                result_matrix[i * p + j + 1] += matrix1[i * n + k] * matrix2[k * p + j + 1];
                result_matrix[i * p + j + 2] += matrix1[i * n + k] * matrix2[k * p + j + 2];
                result_matrix[i * p + j + 3] += matrix1[i * n + k] * matrix2[k * p + j + 3];
            }
        }
    }
}
void ijk_1_4_reg(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < P; j += 4)
        {
            /*这里就是使用寄存器存c的结果的四个寄存器，还有a的*/
            double c_00_reg, c_01_reg, c_02_reg, c_03_reg, a_0p_reg;
            c_00_reg = 0.0;
            c_01_reg = 0.0;
            c_02_reg = 0.0;
            c_03_reg = 0.0;
            for (k = 0; k < M; k++)
            {
                a_0p_reg = matrix1[i * n + k];
                c_00_reg += a_0p_reg * matrix2[k * p + j];
                c_01_reg += a_0p_reg * matrix2[k * p + j + 1];
                c_02_reg += a_0p_reg * matrix2[k * p + j + 2];
                c_03_reg += a_0p_reg * matrix2[k * p + j + 3];
            }
            result_matrix[i * p + j] += c_00_reg;
            result_matrix[i * p + j + 1] += c_01_reg;
            result_matrix[i * p + j + 2] += c_02_reg;
            result_matrix[i * p + j + 3] += c_03_reg;
        }
    }
}
// 用指针寻找B的地址
void ijk_1_4_reg_ptb(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < P; j += 4)
        {
            /*这里就是使用寄存器存c的结果的四个寄存器，还有a的*/
            double c_00_reg, c_01_reg, c_02_reg, c_03_reg, a_0p_reg;
            c_00_reg = 0.0;
            c_01_reg = 0.0;
            c_02_reg = 0.0;
            c_03_reg = 0.0;

            double *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;
            bp0_pntr = &matrix2[0 * p + j];
            bp1_pntr = &matrix2[0 * p + j + 1];
            bp2_pntr = &matrix2[0 * p + j + 2];
            bp3_pntr = &matrix2[0 * p + j + 3];
            for (k = 0; k < M; k++)
            {
                a_0p_reg = matrix1[i * n + k];
                c_00_reg += a_0p_reg * *bp0_pntr;
                bp0_pntr += p;
                c_01_reg += a_0p_reg * *bp1_pntr;
                bp1_pntr += p;
                c_02_reg += a_0p_reg * *bp2_pntr;
                bp2_pntr += p;
                c_03_reg += a_0p_reg * *bp3_pntr;
                bp3_pntr += p;
            }
            result_matrix[i * p + j] += c_00_reg;
            result_matrix[i * p + j + 1] += c_01_reg;
            result_matrix[i * p + j + 2] += c_02_reg;
            result_matrix[i * p + j + 3] += c_03_reg;
        }
    }
}
// 对i和t同时做unroll
void ijk_4_4(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i += UNROLL)
    {
        for (j = 0; j < P; j += UNROLL)
        {
            // 这里实现一行乘四列
            for (k = 0; k < M; k++)
            {
                // 第一行乘四列
                result_matrix[i * p + j] += matrix1[i * n + k] * matrix2[k * p + j];
                result_matrix[i * p + j + 1] += matrix1[i * n + k] * matrix2[k * p + j + 1];
                result_matrix[i * p + j + 2] += matrix1[i * n + k] * matrix2[k * p + j + 2];
                result_matrix[i * p + j + 3] += matrix1[i * n + k] * matrix2[k * p + j + 3];
                // 第二行乘四列
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 1) * p + j + 1] += matrix1[(i + 1) * n + k] * matrix2[k * p + j + 1];
                result_matrix[(i + 1) * p + j + 2] += matrix1[(i + 1) * n + k] * matrix2[k * p + j + 2];
                result_matrix[(i + 1) * p + j + 3] += matrix1[(i + 1) * n + k] * matrix2[k * p + j + 3];
                // 第三行
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 2) * p + j + 1] += matrix1[(i + 2) * n + k] * matrix2[k * p + j + 1];
                result_matrix[(i + 2) * p + j + 2] += matrix1[(i + 2) * n + k] * matrix2[k * p + j + 2];
                result_matrix[(i + 2) * p + j + 3] += matrix1[(i + 2) * n + k] * matrix2[k * p + j + 3];
                // 第四行
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 3) * p + j + 1] += matrix1[(i + 3) * n + k] * matrix2[k * p + j + 1];
                result_matrix[(i + 3) * p + j + 2] += matrix1[(i + 3) * n + k] * matrix2[k * p + j + 2];
                result_matrix[(i + 3) * p + j + 3] += matrix1[(i + 3) * n + k] * matrix2[k * p + j + 3];
            }
        }
    }
}
// 使用向量寄存器
void ijk_4_4_reg(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i += UNROLL)
    {
        for (j = 0; j < P; j += UNROLL)
        {
            /*同样的原理对c的4*4方阵进行储存*/
            double
                c_00_reg,
                c_01_reg, c_02_reg, c_03_reg,
                c_10_reg, c_11_reg, c_12_reg, c_13_reg,
                c_20_reg, c_21_reg, c_22_reg, c_23_reg,
                c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                a_0p_reg,
                a_1p_reg,
                a_2p_reg,
                a_3p_reg;

            // 新的循环要初始化
            c_00_reg = 0;
            c_01_reg = 0;
            c_02_reg = 0;
            c_03_reg = 0;
            c_10_reg = 0;
            c_11_reg = 0;
            c_12_reg = 0;
            c_13_reg = 0;
            c_20_reg = 0;
            c_21_reg = 0;
            c_22_reg = 0;
            c_23_reg = 0;
            c_30_reg = 0;
            c_31_reg = 0;
            c_32_reg = 0;
            c_33_reg = 0;
            for (k = 0; k < M; k++)
            {
                // a矩阵拿元素
                a_0p_reg = matrix1[i * n + k];
                a_1p_reg = matrix1[(i + 1) * n + k];
                a_2p_reg = matrix1[(i + 2) * n + k];
                a_3p_reg = matrix1[(i + 3) * n + k];

                // 第一行乘四列
                c_00_reg += a_0p_reg * matrix2[k * p + j];
                c_01_reg += a_0p_reg * matrix2[k * p + j + 1];
                c_02_reg += a_0p_reg * matrix2[k * p + j + 2];
                c_03_reg += a_0p_reg * matrix2[k * p + j + 3];
                // 第二行乘四列
                c_10_reg += a_1p_reg * matrix2[k * p + j];
                c_11_reg += a_1p_reg * matrix2[k * p + j + 1];
                c_12_reg += a_1p_reg * matrix2[k * p + j + 2];
                c_13_reg += a_1p_reg * matrix2[k * p + j + 3];
                // 第三行
                c_20_reg += a_2p_reg * matrix2[k * p + j];
                c_21_reg += a_2p_reg * matrix2[k * p + j + 1];
                c_22_reg += a_2p_reg * matrix2[k * p + j + 2];
                c_23_reg += a_2p_reg * matrix2[k * p + j + 3];
                // 第四行
                c_30_reg += a_3p_reg * matrix2[k * p + j];
                c_31_reg += a_3p_reg * matrix2[k * p + j + 1];
                c_32_reg += a_3p_reg * matrix2[k * p + j + 2];
                c_33_reg += a_3p_reg * matrix2[k * p + j + 3];
            }
            // // 结束完循环要累加
            result_matrix[i * p + j] += c_00_reg;
            result_matrix[i * p + j + 1] = c_01_reg;
            result_matrix[i * p + j + 2] += c_02_reg;
            result_matrix[i * p + j + 3] += c_03_reg;

            result_matrix[(i + 1) * p + j] += c_10_reg;
            result_matrix[(i + 1) * p + j + 1] += c_11_reg;
            result_matrix[(i + 1) * p + j + 2] += c_12_reg;
            result_matrix[(i + 1) * p + j + 3] += c_13_reg;

            result_matrix[(i + 2) * p + j] += c_20_reg;
            result_matrix[(i + 2) * p + j + 1] += c_21_reg;
            result_matrix[(i + 2) * p + j + 2] += c_22_reg;
            result_matrix[(i + 2) * p + j + 3] += c_23_reg;

            result_matrix[(i + 3) * p + j] += c_30_reg;
            result_matrix[(i + 3) * p + j + 1] += c_31_reg;
            result_matrix[(i + 3) * p + j + 2] += c_32_reg;
            result_matrix[(i + 3) * p + j + 3] += c_33_reg;
        }
    }
}

// 使用向量寄存器,并且对b使用指针
void ijk_4_4_reg_ptb(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i += UNROLL)
    {
        for (j = 0; j < P; j += UNROLL)
        {
            /*同样的原理对c的4*4方阵进行储存*/
            double
                c_00_reg,
                c_01_reg, c_02_reg, c_03_reg,
                c_10_reg, c_11_reg, c_12_reg, c_13_reg,
                c_20_reg, c_21_reg, c_22_reg, c_23_reg,
                c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                a_0p_reg,
                a_1p_reg,
                a_2p_reg,
                a_3p_reg;

            // 新的循环要初始化
            c_00_reg = 0;
            c_01_reg = 0;
            c_02_reg = 0;
            c_03_reg = 0;
            c_10_reg = 0;
            c_11_reg = 0;
            c_12_reg = 0;
            c_13_reg = 0;
            c_20_reg = 0;
            c_21_reg = 0;
            c_22_reg = 0;
            c_23_reg = 0;
            c_30_reg = 0;
            c_31_reg = 0;
            c_32_reg = 0;
            c_33_reg = 0;

            double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
            b_p0_pntr = &matrix2[0 * p + j];
            b_p1_pntr = &matrix2[0 * p + j + 1];
            b_p2_pntr = &matrix2[0 * p + j + 2];
            b_p3_pntr = &matrix2[0 * p + j + 3];

            for (k = 0; k < M; k++)
            {
                // a矩阵拿元素
                a_0p_reg = matrix1[i * n + k];
                a_1p_reg = matrix1[(i + 1) * n + k];
                a_2p_reg = matrix1[(i + 2) * n + k];
                a_3p_reg = matrix1[(i + 3) * n + k];

                // 第一行乘四列
                c_00_reg += a_0p_reg * *b_p0_pntr;
                c_01_reg += a_0p_reg * *b_p1_pntr;
                c_02_reg += a_0p_reg * *b_p2_pntr;
                c_03_reg += a_0p_reg * *b_p3_pntr;
                // 第二行乘四列
                c_10_reg += a_1p_reg * *b_p0_pntr;
                c_11_reg += a_1p_reg * *b_p1_pntr;
                c_12_reg += a_1p_reg * *b_p2_pntr;
                c_13_reg += a_1p_reg * *b_p3_pntr;
                // 第三行
                c_20_reg += a_2p_reg * *b_p0_pntr;
                c_21_reg += a_2p_reg * *b_p1_pntr;
                c_22_reg += a_2p_reg * *b_p2_pntr;
                c_23_reg += a_2p_reg * *b_p3_pntr;
                // 第四行
                c_30_reg += a_3p_reg * *b_p0_pntr;
                c_31_reg += a_3p_reg * *b_p1_pntr;
                c_32_reg += a_3p_reg * *b_p2_pntr;
                c_33_reg += a_3p_reg * *b_p3_pntr;

                // 用完b就加
                b_p0_pntr += p;
                b_p1_pntr += p;
                b_p2_pntr += p;
                b_p3_pntr += p;
            }
            // // 结束完循环要累加
            result_matrix[i * p + j] += c_00_reg;
            result_matrix[i * p + j + 1] = c_01_reg;
            result_matrix[i * p + j + 2] += c_02_reg;
            result_matrix[i * p + j + 3] += c_03_reg;

            result_matrix[(i + 1) * p + j] += c_10_reg;
            result_matrix[(i + 1) * p + j + 1] += c_11_reg;
            result_matrix[(i + 1) * p + j + 2] += c_12_reg;
            result_matrix[(i + 1) * p + j + 3] += c_13_reg;

            result_matrix[(i + 2) * p + j] += c_20_reg;
            result_matrix[(i + 2) * p + j + 1] += c_21_reg;
            result_matrix[(i + 2) * p + j + 2] += c_22_reg;
            result_matrix[(i + 2) * p + j + 3] += c_23_reg;

            result_matrix[(i + 3) * p + j] += c_30_reg;
            result_matrix[(i + 3) * p + j + 1] += c_31_reg;
            result_matrix[(i + 3) * p + j + 2] += c_32_reg;
            result_matrix[(i + 3) * p + j + 3] += c_33_reg;
        }
    }
}

// 使用向量寄存器,并且对b使用register
void ijk_4_4_reg_regb(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i += UNROLL)
    {
        for (j = 0; j < P; j += UNROLL)
        {
            /*同样的原理对c的4*4方阵进行储存*/
            double
                c_00_reg,
                c_01_reg, c_02_reg, c_03_reg,
                c_10_reg, c_11_reg, c_12_reg, c_13_reg,
                c_20_reg, c_21_reg, c_22_reg, c_23_reg,
                c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                a_0p_reg,
                a_1p_reg,
                a_2p_reg,
                a_3p_reg,
                b_p0_reg,
                b_p1_reg,
                b_p2_reg,
                b_p3_reg;

            // 新的循环要初始化
            c_00_reg = 0;
            c_01_reg = 0;
            c_02_reg = 0;
            c_03_reg = 0;
            c_10_reg = 0;
            c_11_reg = 0;
            c_12_reg = 0;
            c_13_reg = 0;
            c_20_reg = 0;
            c_21_reg = 0;
            c_22_reg = 0;
            c_23_reg = 0;
            c_30_reg = 0;
            c_31_reg = 0;
            c_32_reg = 0;
            c_33_reg = 0;

            double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
            b_p0_pntr = &matrix2[0 * p + j];
            b_p1_pntr = &matrix2[0 * p + j + 1];
            b_p2_pntr = &matrix2[0 * p + j + 2];
            b_p3_pntr = &matrix2[0 * p + j + 3];

            for (k = 0; k < M; k++)
            {
                b_p0_reg = *b_p0_pntr;
                b_p1_reg = *b_p1_pntr;
                b_p2_reg = *b_p2_pntr;
                b_p3_reg = *b_p3_pntr;
                // a矩阵拿元素
                a_0p_reg = matrix1[i * n + k];
                a_1p_reg = matrix1[(i + 1) * n + k];
                a_2p_reg = matrix1[(i + 2) * n + k];
                a_3p_reg = matrix1[(i + 3) * n + k];

                // 第一行乘四列
                c_00_reg += a_0p_reg * b_p0_reg;
                c_01_reg += a_0p_reg * b_p1_reg;
                c_02_reg += a_0p_reg * b_p2_reg;
                c_03_reg += a_0p_reg * b_p3_reg;
                // 第二行乘四列
                c_10_reg += a_1p_reg * b_p0_reg;
                c_11_reg += a_1p_reg * b_p1_reg;
                c_12_reg += a_1p_reg * b_p2_reg;
                c_13_reg += a_1p_reg * b_p3_reg;
                // 第三行
                c_20_reg += a_2p_reg * b_p0_reg;
                c_21_reg += a_2p_reg * b_p1_reg;
                c_22_reg += a_2p_reg * b_p2_reg;
                c_23_reg += a_2p_reg * b_p3_reg;
                // 第四行
                c_30_reg += a_3p_reg * b_p0_reg;
                c_31_reg += a_3p_reg * b_p1_reg;
                c_32_reg += a_3p_reg * b_p2_reg;
                c_33_reg += a_3p_reg * b_p3_reg;

                // 用完b就加
                b_p0_pntr += p;
                b_p1_pntr += p;
                b_p2_pntr += p;
                b_p3_pntr += p;
            }
            // // 结束完循环要累加
            result_matrix[i * p + j] += c_00_reg;
            result_matrix[i * p + j + 1] = c_01_reg;
            result_matrix[i * p + j + 2] += c_02_reg;
            result_matrix[i * p + j + 3] += c_03_reg;

            result_matrix[(i + 1) * p + j] += c_10_reg;
            result_matrix[(i + 1) * p + j + 1] += c_11_reg;
            result_matrix[(i + 1) * p + j + 2] += c_12_reg;
            result_matrix[(i + 1) * p + j + 3] += c_13_reg;

            result_matrix[(i + 2) * p + j] += c_20_reg;
            result_matrix[(i + 2) * p + j + 1] += c_21_reg;
            result_matrix[(i + 2) * p + j + 2] += c_22_reg;
            result_matrix[(i + 2) * p + j + 3] += c_23_reg;

            result_matrix[(i + 3) * p + j] += c_30_reg;
            result_matrix[(i + 3) * p + j + 1] += c_31_reg;
            result_matrix[(i + 3) * p + j + 2] += c_32_reg;
            result_matrix[(i + 3) * p + j + 3] += c_33_reg;
        }
    }
}

// 使用向量寄存器,并且对b使用register
void ijk_4_4_reg_regb_twoline(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i += UNROLL)
    {
        for (j = 0; j < P; j += UNROLL)
        {

            /*同样的原理对c的4*4方阵进行储存*/
            double
                c_00_reg,
                c_01_reg, c_02_reg, c_03_reg,
                c_10_reg, c_11_reg, c_12_reg, c_13_reg,
                c_20_reg, c_21_reg, c_22_reg, c_23_reg,
                c_30_reg, c_31_reg, c_32_reg, c_33_reg,
                a_0p_reg,
                a_1p_reg,
                a_2p_reg,
                a_3p_reg,
                b_p0_reg,
                b_p1_reg,
                b_p2_reg,
                b_p3_reg;

            // 新的循环要初始化
            c_00_reg = 0;
            c_01_reg = 0;
            c_02_reg = 0;
            c_03_reg = 0;
            c_10_reg = 0;
            c_11_reg = 0;
            c_12_reg = 0;
            c_13_reg = 0;
            c_20_reg = 0;
            c_21_reg = 0;
            c_22_reg = 0;
            c_23_reg = 0;
            c_30_reg = 0;
            c_31_reg = 0;
            c_32_reg = 0;
            c_33_reg = 0;

            double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
            b_p0_pntr = &matrix2[0 * p + j];
            b_p1_pntr = &matrix2[0 * p + j + 1];
            b_p2_pntr = &matrix2[0 * p + j + 2];
            b_p3_pntr = &matrix2[0 * p + j + 3];

            for (k = 0; k < M; k++)
            {
                b_p0_reg = *b_p0_pntr;
                b_p1_reg = *b_p1_pntr;
                b_p2_reg = *b_p2_pntr;
                b_p3_reg = *b_p3_pntr;
                // a矩阵拿元素
                a_0p_reg = matrix1[i * n + k];
                a_1p_reg = matrix1[(i + 1) * n + k];
                a_2p_reg = matrix1[(i + 2) * n + k];
                a_3p_reg = matrix1[(i + 3) * n + k];

                c_00_reg += a_0p_reg * b_p0_reg;
                c_10_reg += a_1p_reg * b_p0_reg;

                c_01_reg += a_0p_reg * b_p1_reg;
                c_11_reg += a_1p_reg * b_p1_reg;

                c_02_reg += a_0p_reg * b_p2_reg;
                c_12_reg += a_1p_reg * b_p2_reg;

                c_03_reg += a_0p_reg * b_p3_reg;
                c_13_reg += a_1p_reg * b_p3_reg;

                c_20_reg += a_2p_reg * b_p0_reg;
                c_30_reg += a_3p_reg * b_p0_reg;

                c_21_reg += a_2p_reg * b_p1_reg;
                c_31_reg += a_3p_reg * b_p1_reg;

                c_22_reg += a_2p_reg * b_p2_reg;
                c_32_reg += a_3p_reg * b_p2_reg;

                c_23_reg += a_2p_reg * b_p3_reg;
                c_33_reg += a_3p_reg * b_p3_reg;

                // 用完b就加
                b_p0_pntr += p;
                b_p1_pntr += p;
                b_p2_pntr += p;
                b_p3_pntr += p;
            }

            // // 结束完循环要累加
            result_matrix[i * p + j] += c_00_reg;
            result_matrix[i * p + j + 1] = c_01_reg;
            result_matrix[i * p + j + 2] += c_02_reg;
            result_matrix[i * p + j + 3] += c_03_reg;

            result_matrix[(i + 1) * p + j] += c_10_reg;
            result_matrix[(i + 1) * p + j + 1] += c_11_reg;
            result_matrix[(i + 1) * p + j + 2] += c_12_reg;
            result_matrix[(i + 1) * p + j + 3] += c_13_reg;

            result_matrix[(i + 2) * p + j] += c_20_reg;
            result_matrix[(i + 2) * p + j + 1] += c_21_reg;
            result_matrix[(i + 2) * p + j + 2] += c_22_reg;
            result_matrix[(i + 2) * p + j + 3] += c_23_reg;

            result_matrix[(i + 3) * p + j] += c_30_reg;
            result_matrix[(i + 3) * p + j + 1] += c_31_reg;
            result_matrix[(i + 3) * p + j + 2] += c_32_reg;
            result_matrix[(i + 3) * p + j + 3] += c_33_reg;
        }
    }
}

// 使用向量寄存器,并且对b使用register
void ijk_4_4_reg_regb_sse3(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i += UNROLL)
    {
        for (j = 0; j < P; j += UNROLL)
        {
            __m128d c_00_c_01_vec, c_02_c_03_vec, c_10_c_11_vec, c_12_c_13_vec,
                c_20_c_21_vec, c_22_c_23_vec, c_30_c_31_vec, c_32_c_33_vec,
                a_0p_vec, a_1p_vec, a_2p_vec, a_3p_vec,
                b_p0_b_p1_vec, b_p2_b_p3_vec;

            double *b_p0_pntr = NULL, *b_p1_pntr = NULL, *b_p2_pntr = NULL, *b_p3_pntr = NULL;

            b_p0_pntr = &matrix2[0 * p + j];
            b_p1_pntr = &matrix2[0 * p + j + 1];
            b_p2_pntr = &matrix2[0 * p + j + 2];
            b_p3_pntr = &matrix2[0 * p + j + 3];

            // 还是经典的清空
            c_00_c_01_vec = _mm_setzero_pd();
            c_02_c_03_vec = _mm_setzero_pd();
            c_10_c_11_vec = _mm_setzero_pd();
            c_12_c_13_vec = _mm_setzero_pd();
            c_20_c_21_vec = _mm_setzero_pd();
            c_22_c_23_vec = _mm_setzero_pd();
            c_30_c_31_vec = _mm_setzero_pd();
            c_32_c_33_vec = _mm_setzero_pd();
            for (k = 0; k < M; k++)
            {
                b_p0_b_p1_vec = _mm_loadu_pd((double *)b_p0_pntr);
                b_p2_b_p3_vec = _mm_loadu_pd((double *)b_p2_pntr);

                // a这么拿一次是拿一列
                a_0p_vec = _mm_load1_pd((double *)&matrix1[i * n + k]);
                a_1p_vec = _mm_load1_pd((double *)&matrix1[(i + 1) * n + k]);
                a_2p_vec = _mm_load1_pd((double *)&matrix1[(i + 2) * n + k]);
                a_3p_vec = _mm_load1_pd((double *)&matrix1[(i + 3) * n + k]);

                // _mm_add_pd(c_00_c_01_vec, _mm_mul_pd(a_0p_vec, b_p0_b_p1_vec));
                // _mm_add_pd(c_02_c_03_vec, _mm_mul_pd(a_0p_vec, b_p2_b_p3_vec));

                // _mm_add_pd(c_10_c_11_vec, _mm_mul_pd(a_1p_vec, b_p0_b_p1_vec));
                // _mm_add_pd(c_12_c_13_vec, _mm_mul_pd(a_1p_vec, b_p2_b_p3_vec));

                // _mm_add_pd(c_20_c_21_vec, _mm_mul_pd(a_2p_vec, b_p0_b_p1_vec));
                // _mm_add_pd(c_22_c_23_vec, _mm_mul_pd(a_2p_vec, b_p2_b_p3_vec));

                // _mm_add_pd(c_30_c_31_vec, _mm_mul_pd(a_3p_vec, b_p0_b_p1_vec));
                // _mm_add_pd(c_32_c_33_vec, _mm_mul_pd(a_3p_vec, b_p2_b_p3_vec));
                c_00_c_01_vec += a_0p_vec * b_p0_b_p1_vec;
                c_02_c_03_vec += a_0p_vec * b_p2_b_p3_vec;

                c_10_c_11_vec += a_1p_vec * b_p0_b_p1_vec;
                c_12_c_13_vec += a_1p_vec * b_p2_b_p3_vec;

                c_20_c_21_vec += a_2p_vec * b_p0_b_p1_vec;
                c_22_c_23_vec += a_2p_vec * b_p2_b_p3_vec;

                c_30_c_31_vec += a_3p_vec * b_p0_b_p1_vec;
                c_32_c_33_vec += a_3p_vec * b_p2_b_p3_vec;
                // 用完b就加
                b_p0_pntr += p;
                b_p1_pntr += p;
                b_p2_pntr += p;
                b_p3_pntr += p;
            }
            // 更新result_matrix
            // 更新result_matrix
            result_matrix[i * p + j] += c_00_c_01_vec[0];
            result_matrix[i * p + j + 1] = c_00_c_01_vec[1];
            result_matrix[i * p + j + 2] += c_02_c_03_vec[0];
            result_matrix[i * p + j + 3] += c_02_c_03_vec[1];

            result_matrix[(i + 1) * p + j] += c_10_c_11_vec[0];
            result_matrix[(i + 1) * p + j + 1] += c_10_c_11_vec[1];
            result_matrix[(i + 1) * p + j + 2] += c_12_c_13_vec[0];
            result_matrix[(i + 1) * p + j + 3] += c_12_c_13_vec[1];

            result_matrix[(i + 2) * p + j] += c_20_c_21_vec[0];
            result_matrix[(i + 2) * p + j + 1] += c_20_c_21_vec[1];
            result_matrix[(i + 2) * p + j + 2] += c_22_c_23_vec[0];
            result_matrix[(i + 2) * p + j + 3] += c_22_c_23_vec[1];

            result_matrix[(i + 3) * p + j] += c_30_c_31_vec[0];
            result_matrix[(i + 3) * p + j + 1] += c_30_c_31_vec[1];
            result_matrix[(i + 3) * p + j + 2] += c_32_c_33_vec[0];
            result_matrix[(i + 3) * p + j + 3] += c_32_c_33_vec[1];
        }
    }
}

void do_block(double *matrix1, double *matrix2, double *result_matrix, int i_o, int j_o, int k_o, int n, int p)
{
    int i, j, k;
    for (i = 0; i < BLOCKSIZE; i += UNROLL)
    {
        for (j = 0; j < BLOCKSIZE; j += UNROLL)
        {
            __m128d c_00_c_01_vec, c_02_c_03_vec, c_10_c_11_vec, c_12_c_13_vec,
                c_20_c_21_vec, c_22_c_23_vec, c_30_c_31_vec, c_32_c_33_vec,
                a_0p_vec, a_1p_vec, a_2p_vec, a_3p_vec,
                b_p0_b_p1_vec, b_p2_b_p3_vec;

            double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
            b_p0_pntr = &matrix2[0 * p + j];
            b_p1_pntr = &matrix2[0 * p + j + 1];
            b_p2_pntr = &matrix2[0 * p + j + 2];
            b_p3_pntr = &matrix2[0 * p + j + 3];

            // 还是经典的清空
            c_00_c_01_vec = _mm_setzero_pd();
            c_02_c_03_vec = _mm_setzero_pd();
            c_10_c_11_vec = _mm_setzero_pd();
            c_12_c_13_vec = _mm_setzero_pd();
            c_20_c_21_vec = _mm_setzero_pd();
            c_22_c_23_vec = _mm_setzero_pd();
            c_30_c_31_vec = _mm_setzero_pd();
            c_32_c_33_vec = _mm_setzero_pd();
            for (k = 0; k < BLOCKSIZE; k++)
            {
                b_p0_b_p1_vec = _mm_loadu_pd((double *)b_p0_pntr);
                b_p2_b_p3_vec = _mm_loadu_pd((double *)b_p2_pntr);

                // a这么拿一次是拿一列
                a_0p_vec = _mm_load1_pd((double *)&matrix1[i * n + k]);
                a_1p_vec = _mm_load1_pd((double *)&matrix1[(i + 1) * n + k]);
                a_2p_vec = _mm_load1_pd((double *)&matrix1[(i + 2) * n + k]);
                a_3p_vec = _mm_load1_pd((double *)&matrix1[(i + 3) * n + k]);

                c_00_c_01_vec += a_0p_vec * b_p0_b_p1_vec;
                c_02_c_03_vec += a_0p_vec * b_p2_b_p3_vec;

                c_10_c_11_vec += a_1p_vec * b_p0_b_p1_vec;
                c_12_c_13_vec += a_1p_vec * b_p2_b_p3_vec;

                c_20_c_21_vec += a_2p_vec * b_p0_b_p1_vec;
                c_22_c_23_vec += a_2p_vec * b_p2_b_p3_vec;

                c_30_c_31_vec += a_3p_vec * b_p0_b_p1_vec;
                c_32_c_33_vec += a_3p_vec * b_p2_b_p3_vec;
                // 用完b就加
                b_p0_pntr += p;
                b_p1_pntr += p;
                b_p2_pntr += p;
                b_p3_pntr += p;
            }
            // 更新result_matrix
            // 更新result_matrix
            result_matrix[i * p + j] += c_00_c_01_vec[0];
            result_matrix[i * p + j + 1] += c_00_c_01_vec[1];
            result_matrix[i * p + j + 2] += c_02_c_03_vec[0];
            result_matrix[i * p + j + 3] += c_02_c_03_vec[1];

            result_matrix[(i + 1) * p + j] += c_10_c_11_vec[0];
            result_matrix[(i + 1) * p + j + 1] += c_10_c_11_vec[1];
            result_matrix[(i + 1) * p + j + 2] += c_12_c_13_vec[0];
            result_matrix[(i + 1) * p + j + 3] += c_12_c_13_vec[1];

            result_matrix[(i + 2) * p + j] += c_20_c_21_vec[0];
            result_matrix[(i + 2) * p + j + 1] += c_20_c_21_vec[1];
            result_matrix[(i + 2) * p + j + 2] += c_22_c_23_vec[0];
            result_matrix[(i + 2) * p + j + 3] += c_22_c_23_vec[1];

            result_matrix[(i + 3) * p + j] += c_30_c_31_vec[0];
            result_matrix[(i + 3) * p + j + 1] += c_30_c_31_vec[1];
            result_matrix[(i + 3) * p + j + 2] += c_32_c_33_vec[0];
            result_matrix[(i + 3) * p + j + 3] += c_32_c_33_vec[1];
        }
    }
}
// 这里要把4*M,M*4分成更小的运算单位，因为M过大，L1cache还是粗存不了
void ijk_4_4_reg_regb_sse3_block(double *matrxi1, double *matrix2, double *result_matrix, int n, int p)
{
    for (int i_o = 0; i_o < N; i_o += BLOCKSIZE)
    {
        for (int j_o = 0; j_o < P; j_o += BLOCKSIZE)
        {
            for (int k_o = 0; k_o < M; k_o += BLOCKSIZE)
            {
                do_block(&matrxi1[i_o * n + k_o], &matrix2[k_o * p + j_o], &result_matrix[i_o * p + j_o], i_o, j_o, k_o, n, p);
            }
        }
    }
}
// 实现最关键的打包函数
void PackMaxtrix1(double *matrix1, double *matrix1_to, int n, int p) // p是BLOCKSIZE
{
    /*第一个是原来矩阵起始的地址，第二个是一维数组储存，n是A的列，p是B的列*/
    int i;
    double *a_ij_pntr_0 = &matrix1[0],
           *a_ij_pntr_1 = &matrix1[1 * n],
           *a_ij_pntr_2 = &matrix1[2 * n],
           *a_ij_pntr_3 = &matrix1[3 * n];
    for (i = 0; i < p; i++)
    {
        *matrix1_to++ = *a_ij_pntr_0++;
        *matrix1_to++ = *a_ij_pntr_1++;
        *matrix1_to++ = *a_ij_pntr_2++;
        *matrix1_to++ = *a_ij_pntr_3++;
    }
}
void ijk_4_4_reg_regb_sse3_packa(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    static double PackA[M * N];
    for (i = 0; i < N; i += UNROLL)
    {
        // 打包A,这里是没问题的
        PackMaxtrix1(&matrix1[i * n], &PackA[i * n], N, P);
        for (j = 0; j < P; j += UNROLL)
        {
            __m128d
                c_00_c_01_vec,
                c_02_c_03_vec, c_10_c_11_vec, c_12_c_13_vec,
                c_20_c_21_vec, c_22_c_23_vec, c_30_c_31_vec, c_32_c_33_vec,
                a_0p_vec, a_1p_vec, a_2p_vec, a_3p_vec,
                b_p0_b_p1_vec, b_p2_b_p3_vec;

            double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
            b_p0_pntr = &matrix2[0 * p + j];
            b_p1_pntr = &matrix2[0 * p + j + 1];
            b_p2_pntr = &matrix2[0 * p + j + 2];
            b_p3_pntr = &matrix2[0 * p + j + 3];

            // 还是经典的清空
            c_00_c_01_vec = _mm_setzero_pd();
            c_02_c_03_vec = _mm_setzero_pd();
            c_10_c_11_vec = _mm_setzero_pd();
            c_12_c_13_vec = _mm_setzero_pd();
            c_20_c_21_vec = _mm_setzero_pd();
            c_22_c_23_vec = _mm_setzero_pd();
            c_30_c_31_vec = _mm_setzero_pd();
            c_32_c_33_vec = _mm_setzero_pd();
            for (k = 0; k < M; k++)
            {
                // 这里才能进行打包,最开始进行一次打包

                b_p0_b_p1_vec = _mm_loadu_pd((double *)b_p0_pntr);
                b_p2_b_p3_vec = _mm_loadu_pd((double *)b_p2_pntr);

                // a这么拿一次是拿一列
                a_0p_vec = _mm_load1_pd((double *)&PackA[i * n + k * 4]);
                a_1p_vec = _mm_load1_pd((double *)&PackA[(i * n + 1) + k * 4]);
                a_2p_vec = _mm_load1_pd((double *)&PackA[(i * n + 2) + k * 4]);
                a_3p_vec = _mm_load1_pd((double *)&PackA[(i * n + 3) + k * 4]);
                c_00_c_01_vec += a_0p_vec * b_p0_b_p1_vec;
                c_02_c_03_vec += a_0p_vec * b_p2_b_p3_vec;

                c_10_c_11_vec += a_1p_vec * b_p0_b_p1_vec;
                c_12_c_13_vec += a_1p_vec * b_p2_b_p3_vec;

                c_20_c_21_vec += a_2p_vec * b_p0_b_p1_vec;
                c_22_c_23_vec += a_2p_vec * b_p2_b_p3_vec;

                c_30_c_31_vec += a_3p_vec * b_p0_b_p1_vec;
                c_32_c_33_vec += a_3p_vec * b_p2_b_p3_vec;
                // 用完b就加
                b_p0_pntr += p;
                b_p1_pntr += p;
                b_p2_pntr += p;
                b_p3_pntr += p;
            }
            // 更新result_matrix
            // 更新result_matrix
            result_matrix[i * p + j] += c_00_c_01_vec[0];
            result_matrix[i * p + j + 1] = c_00_c_01_vec[1];
            result_matrix[i * p + j + 2] += c_02_c_03_vec[0];
            result_matrix[i * p + j + 3] += c_02_c_03_vec[1];

            result_matrix[(i + 1) * p + j] += c_10_c_11_vec[0];
            result_matrix[(i + 1) * p + j + 1] += c_10_c_11_vec[1];
            result_matrix[(i + 1) * p + j + 2] += c_12_c_13_vec[0];
            result_matrix[(i + 1) * p + j + 3] += c_12_c_13_vec[1];

            result_matrix[(i + 2) * p + j] += c_20_c_21_vec[0];
            result_matrix[(i + 2) * p + j + 1] += c_20_c_21_vec[1];
            result_matrix[(i + 2) * p + j + 2] += c_22_c_23_vec[0];
            result_matrix[(i + 2) * p + j + 3] += c_22_c_23_vec[1];

            result_matrix[(i + 3) * p + j] += c_30_c_31_vec[0];
            result_matrix[(i + 3) * p + j + 1] += c_30_c_31_vec[1];
            result_matrix[(i + 3) * p + j + 2] += c_32_c_33_vec[0];
            result_matrix[(i + 3) * p + j + 3] += c_32_c_33_vec[1];
        }
    }
}
void do_block_packA(double *matrix1, double *matrix2, double *result_matrix, int i_o, int j_o, int k_o, int n, int p, bool first_use_a)
{
    int i, j, k;
    static double PackA[BLOCKSIZE * BLOCKSIZE];
    for (i = 0; i < BLOCKSIZE; i += UNROLL)
    {
        if (first_use_a)
        {
            PackMaxtrix1(&matrix1[i * n], &PackA[i * BLOCKSIZE], n, BLOCKSIZE);
        }

        for (j = 0; j < BLOCKSIZE; j += UNROLL)
        {
            __m128d c_00_c_01_vec, c_02_c_03_vec, c_10_c_11_vec, c_12_c_13_vec,
                c_20_c_21_vec, c_22_c_23_vec, c_30_c_31_vec, c_32_c_33_vec,
                a_0p_vec, a_1p_vec, a_2p_vec, a_3p_vec,
                b_p0_b_p1_vec, b_p2_b_p3_vec;

            double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
            b_p0_pntr = &matrix2[0 * p + j];
            b_p1_pntr = &matrix2[0 * p + j + 1];
            b_p2_pntr = &matrix2[0 * p + j + 2];
            b_p3_pntr = &matrix2[0 * p + j + 3];

            // 还是经典的清空
            c_00_c_01_vec = _mm_setzero_pd();
            c_02_c_03_vec = _mm_setzero_pd();
            c_10_c_11_vec = _mm_setzero_pd();
            c_12_c_13_vec = _mm_setzero_pd();
            c_20_c_21_vec = _mm_setzero_pd();
            c_22_c_23_vec = _mm_setzero_pd();
            c_30_c_31_vec = _mm_setzero_pd();
            c_32_c_33_vec = _mm_setzero_pd();
            for (k = 0; k < BLOCKSIZE; k++)
            {
                b_p0_b_p1_vec = _mm_loadu_pd((double *)b_p0_pntr);
                b_p2_b_p3_vec = _mm_loadu_pd((double *)b_p2_pntr);

                // a这么拿一次是拿一列
                a_0p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4]);
                a_1p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 1]);
                a_2p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 2]);
                a_3p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 3]);

                c_00_c_01_vec += a_0p_vec * b_p0_b_p1_vec;
                c_02_c_03_vec += a_0p_vec * b_p2_b_p3_vec;

                c_10_c_11_vec += a_1p_vec * b_p0_b_p1_vec;
                c_12_c_13_vec += a_1p_vec * b_p2_b_p3_vec;

                c_20_c_21_vec += a_2p_vec * b_p0_b_p1_vec;
                c_22_c_23_vec += a_2p_vec * b_p2_b_p3_vec;

                c_30_c_31_vec += a_3p_vec * b_p0_b_p1_vec;
                c_32_c_33_vec += a_3p_vec * b_p2_b_p3_vec;
                // 用完b就加
                b_p0_pntr += p;
                b_p1_pntr += p;
                b_p2_pntr += p;
                b_p3_pntr += p;
            }
            // 更新result_matrix
            // 更新result_matrix
            result_matrix[i * p + j] += c_00_c_01_vec[0];
            result_matrix[i * p + j + 1] += c_00_c_01_vec[1];
            result_matrix[i * p + j + 2] += c_02_c_03_vec[0];
            result_matrix[i * p + j + 3] += c_02_c_03_vec[1];

            result_matrix[(i + 1) * p + j] += c_10_c_11_vec[0];
            result_matrix[(i + 1) * p + j + 1] += c_10_c_11_vec[1];
            result_matrix[(i + 1) * p + j + 2] += c_12_c_13_vec[0];
            result_matrix[(i + 1) * p + j + 3] += c_12_c_13_vec[1];

            result_matrix[(i + 2) * p + j] += c_20_c_21_vec[0];
            result_matrix[(i + 2) * p + j + 1] += c_20_c_21_vec[1];
            result_matrix[(i + 2) * p + j + 2] += c_22_c_23_vec[0];
            result_matrix[(i + 2) * p + j + 3] += c_22_c_23_vec[1];

            result_matrix[(i + 3) * p + j] += c_30_c_31_vec[0];
            result_matrix[(i + 3) * p + j + 1] += c_30_c_31_vec[1];
            result_matrix[(i + 3) * p + j + 2] += c_32_c_33_vec[0];
            result_matrix[(i + 3) * p + j + 3] += c_32_c_33_vec[1];
        }
    }
}
void ijk_4_4_reg_regb_sse3_block_packa(double *matrxi1, double *matrix2, double *result_matrix, int n, int p)
{
    for (int i_o = 0; i_o < N; i_o += BLOCKSIZE)
    {
        for (int k_o = 0; k_o < M; k_o += BLOCKSIZE)
        {
            for (int j_o = 0; j_o < P; j_o += BLOCKSIZE)
            {
                do_block_packA(&matrxi1[i_o * n + k_o], &matrix2[k_o * p + j_o], &result_matrix[i_o * p + j_o], i_o, j_o, k_o, n, p, j_o == 0);
            }
        }
    }
}
// 打包数组B
void PackMaxtrix2(double *matrix2, double *matrix2_to, int n, int p) // 注意这里的p是一行元素
{
    /*第一个是原来矩阵起始的地址，第二个是一维数组储存，n是A的列，p是B的列*/
    int j;
    // 实际上是BLOCKSIZE
    for (j = 0; j < n; j++)
    {
        double *b_kj_pntr = &matrix2[j * p];

        // 指四个
        *matrix2_to = *b_kj_pntr;
        *(matrix2_to + 1) = *(b_kj_pntr + 1);
        *(matrix2_to + 2) = *(b_kj_pntr + 2);
        *(matrix2_to + 3) = *(b_kj_pntr + 3);
        matrix2_to += 4;
    }
}

void do_block_packAB(double *matrix1, double *matrix2, double *result_matrix, int i_o, int j_o, int k_o, int n, int p, bool first_use_a)
{
    int i, j, k;
    static double PackA[BLOCKSIZE * BLOCKSIZE];
    static double PackB[BLOCKSIZE * BLOCKSIZE];
    for (i = 0; i < BLOCKSIZE; i += UNROLL)
    {
        if (first_use_a)
        {
            PackMaxtrix1(&matrix1[i * n], &PackA[i * BLOCKSIZE], n, BLOCKSIZE); // 修改a结果把b修改了
        }

        for (j = 0; j < BLOCKSIZE; j += UNROLL)
        {
            if (i == 0)
            {
                PackMaxtrix2(&matrix2[0 * p + j], &PackB[j * BLOCKSIZE], BLOCKSIZE, P);
            }

            __m128d c_00_c_01_vec,
                c_02_c_03_vec, c_10_c_11_vec, c_12_c_13_vec,
                c_20_c_21_vec, c_22_c_23_vec, c_30_c_31_vec, c_32_c_33_vec,
                a_0p_vec, a_1p_vec, a_2p_vec, a_3p_vec,
                b_p0_b_p1_vec, b_p2_b_p3_vec;

            double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;
            b_p0_pntr = &PackB[j * BLOCKSIZE];
            b_p1_pntr = &PackB[j * BLOCKSIZE + 1];
            b_p2_pntr = &PackB[j * BLOCKSIZE + 2];
            b_p3_pntr = &PackB[j * BLOCKSIZE + 3];

            // 还是经典的清空
            c_00_c_01_vec = _mm_setzero_pd();
            c_02_c_03_vec = _mm_setzero_pd();
            c_10_c_11_vec = _mm_setzero_pd();
            c_12_c_13_vec = _mm_setzero_pd();
            c_20_c_21_vec = _mm_setzero_pd();
            c_22_c_23_vec = _mm_setzero_pd();
            c_30_c_31_vec = _mm_setzero_pd();
            c_32_c_33_vec = _mm_setzero_pd();
            for (k = 0; k < BLOCKSIZE; k++)
            {
                b_p0_b_p1_vec = _mm_loadu_pd((double *)b_p0_pntr);
                b_p2_b_p3_vec = _mm_loadu_pd((double *)b_p2_pntr);

                // a这么拿一次是拿一列
                a_0p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4]);
                a_1p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 1]);
                a_2p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 2]);
                a_3p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 3]);

                c_00_c_01_vec += a_0p_vec * b_p0_b_p1_vec;
                c_02_c_03_vec += a_0p_vec * b_p2_b_p3_vec;

                c_10_c_11_vec += a_1p_vec * b_p0_b_p1_vec;
                c_12_c_13_vec += a_1p_vec * b_p2_b_p3_vec;

                c_20_c_21_vec += a_2p_vec * b_p0_b_p1_vec;
                c_22_c_23_vec += a_2p_vec * b_p2_b_p3_vec;

                c_30_c_31_vec += a_3p_vec * b_p0_b_p1_vec;
                c_32_c_33_vec += a_3p_vec * b_p2_b_p3_vec;
                // 用完b就加
                b_p0_pntr += 4;
                b_p1_pntr += 4;
                b_p2_pntr += 4;
                b_p3_pntr += 4;
            }
            // 更新result_matrix
            result_matrix[i * p + j] += c_00_c_01_vec[0];
            result_matrix[i * p + j + 1] += c_00_c_01_vec[1];
            result_matrix[i * p + j + 2] += c_02_c_03_vec[0];
            result_matrix[i * p + j + 3] += c_02_c_03_vec[1];

            result_matrix[(i + 1) * p + j] += c_10_c_11_vec[0];
            result_matrix[(i + 1) * p + j + 1] += c_10_c_11_vec[1];
            result_matrix[(i + 1) * p + j + 2] += c_12_c_13_vec[0];
            result_matrix[(i + 1) * p + j + 3] += c_12_c_13_vec[1];

            result_matrix[(i + 2) * p + j] += c_20_c_21_vec[0];
            result_matrix[(i + 2) * p + j + 1] += c_20_c_21_vec[1];
            result_matrix[(i + 2) * p + j + 2] += c_22_c_23_vec[0];
            result_matrix[(i + 2) * p + j + 3] += c_22_c_23_vec[1];

            result_matrix[(i + 3) * p + j] += c_30_c_31_vec[0];
            result_matrix[(i + 3) * p + j + 1] += c_30_c_31_vec[1];
            result_matrix[(i + 3) * p + j + 2] += c_32_c_33_vec[0];
            result_matrix[(i + 3) * p + j + 3] += c_32_c_33_vec[1];
        }
    }
}
void ijk_4_4_reg_regb_sse3_block_packab(double *matrxi1, double *matrix2, double *result_matrix, int n, int p)
{

    for (int i_o = 0; i_o < N; i_o += BLOCKSIZE)
    {
        for (int k_o = 0; k_o < M; k_o += BLOCKSIZE)
        {
            for (int j_o = 0; j_o < P; j_o += BLOCKSIZE)
            {
                do_block_packAB(&matrxi1[i_o * n + k_o], &matrix2[k_o * p + j_o], &result_matrix[i_o * p + j_o], i_o, j_o, k_o, n, p, j_o == 0);
            }
        }
    }
}
// ikj的运算方式
void ikj_maxtrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P])
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (k = 0; k < M; k++)
        {
            for (j = 0; j < P; j++) //  相当于一行乘一行
            {
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}
// ikj 加入unroll
void ikj_1_4(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (k = 0; k < M; k += UNROLL)
        {
            for (j = 0; j < P; j++)
            {
                result_matrix[i * p + j] += matrix1[i * n + k] * matrix2[k * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 3] * matrix2[(k + 3) * p + j];
            }
        }
    }
}
void ikj_4_4(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i += UNROLL)
    {
        for (k = 0; k < M; k += UNROLL)
        {
            for (j = 0; j < P; j++)
            {
                // 其实这个意识一个道理，result用指针，16个是a,四个的是b
                // 第一行乘第一个矩阵=第一行
                result_matrix[i * p + j] += matrix1[i * n + k] * matrix2[k * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 3] * matrix2[(k + 3) * p + j];
                // 第二行乘第二个矩阵
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k + 3] * matrix2[(k + 3) * p + j];
                // 第三行
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k + 3] * matrix2[(k + 3) * p + j];
                // 第四行
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k + 3] * matrix2[(k + 3) * p + j];
            }
        }
    }
}
void ikj_4_4_reg(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i, j, k;
    for (i = 0; i < N; i += UNROLL)
    {
        for (k = 0; k < M; k += UNROLL)
        {

            // 首先是16个a的寄存器
            double
                a_00_reg,
                a_01_reg, a_02_reg, a_03_reg,
                a_10_reg, a_11_reg, a_12_reg, a_13_reg,
                a_20_reg, a_21_reg, a_22_reg, a_23_reg,
                a_30_reg, a_31_reg, a_32_reg, a_33_reg,
                b_k0_reg,
                b_k1_reg,
                b_k2_reg,
                b_k3_reg;

            for (j = 0; j < P; j++)
            {
                // 其实这个意识一个道理，result用指针，16个是a,四个的是b
                // 第一行乘第一个矩阵=第一行
                result_matrix[i * p + j] += matrix1[i * n + k] * matrix2[k * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 3] * matrix2[(k + 3) * p + j];
                // 第二行乘第二个矩阵
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[(i + 1) * p + j] += matrix1[(i + 1) * n + k + 3] * matrix2[(k + 3) * p + j];
                // 第三行
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[(i + 2) * p + j] += matrix1[(i + 2) * n + k + 3] * matrix2[(k + 3) * p + j];
                // 第四行
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k] * matrix2[k * p + j];
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[(i + 3) * p + j] += matrix1[(i + 3) * n + k + 3] * matrix2[(k + 3) * p + j];
            }
        }
    }
}
// 使用寄存器
void ikj_1_4_reg(double *matrix1, double *matrix2, double *result_matrix, int n, int p)
{
    int i = 0, j = 0, k = 0;
    for (i = 0; i < N; i++)
    {
        for (k = 0; k < M; k += UNROLL)
        {
            double a_00_reg, a_01_reg, a_02_reg, a_03_reg;
            double c_ij_reg;
            for (j = 0; j < P; j++)
            {
                result_matrix[i * p + j] += matrix1[i * n + k] * matrix2[k * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 1] * matrix2[(k + 1) * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 2] * matrix2[(k + 2) * p + j];
                result_matrix[i * p + j] += matrix1[i * n + k + 3] * matrix2[(k + 3) * p + j];
            }
        }
    }
}
void do_block_packAB_new(double *matrix1, double *matrix2, double *result_matrix, int i_o, int j_o, int k_o, int n, int p, bool first_use_a, double *PackA, double *PackB)
{
    int i, j, k;
    for (i = 0; i < BLOCKSIZE; i += UNROLL)
    {
        if (first_use_a)
        {
            PackMaxtrix1(&matrix1[i * n], &PackA[i * BLOCKSIZEK], n, BLOCKSIZEK);
        }

        for (j = 0; j < BLOCKSIZE; j += UNROLL)
        {
            if (i == 0) // 这里是打包B
            {
                PackMaxtrix2(&matrix2[0 * p + j], &PackB[j * BLOCKSIZEK], BLOCKSIZEK, P);
            }
            // 这里实际有一条分割线
            __m128d c_00_c_01_vec,
                c_02_c_03_vec, c_10_c_11_vec, c_12_c_13_vec,
                c_20_c_21_vec, c_22_c_23_vec, c_30_c_31_vec, c_32_c_33_vec,
                a_0p_vec, a_1p_vec, a_2p_vec, a_3p_vec,
                b_p0_b_p1_vec, b_p2_b_p3_vec;
            double *b = &PackB[j * BLOCKSIZEK];
            double *a = &PackA[i * BLOCKSIZEK];
            // 还是经典的清空
            c_00_c_01_vec = _mm_setzero_pd();
            c_02_c_03_vec = _mm_setzero_pd();
            c_10_c_11_vec = _mm_setzero_pd();
            c_12_c_13_vec = _mm_setzero_pd();
            c_20_c_21_vec = _mm_setzero_pd();
            c_22_c_23_vec = _mm_setzero_pd();
            c_30_c_31_vec = _mm_setzero_pd();
            c_32_c_33_vec = _mm_setzero_pd();
            for (k = 0; k < BLOCKSIZEK; k++)
            {

                b_p0_b_p1_vec = _mm_loadu_pd((double *)b);
                b_p2_b_p3_vec = _mm_loadu_pd((double *)(b + 2));
                b += 4;
                // a_0p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4]);
                // a_1p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 1]);
                // a_2p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 2]);
                // a_3p_vec = _mm_load1_pd((double *)&PackA[i * BLOCKSIZE + k * 4 + 3]);

                // a这么拿一次是拿一列
                a_0p_vec = _mm_load1_pd((double *)a);
                a_1p_vec = _mm_load1_pd((double *)(a + 1));
                a_2p_vec = _mm_load1_pd((double *)(a + 2));
                a_3p_vec = _mm_load1_pd((double *)(a + 3));
                a += 4;
                c_00_c_01_vec += a_0p_vec * b_p0_b_p1_vec;
                c_02_c_03_vec += a_0p_vec * b_p2_b_p3_vec;

                c_10_c_11_vec += a_1p_vec * b_p0_b_p1_vec;
                c_12_c_13_vec += a_1p_vec * b_p2_b_p3_vec;

                c_20_c_21_vec += a_2p_vec * b_p0_b_p1_vec;
                c_22_c_23_vec += a_2p_vec * b_p2_b_p3_vec;

                c_30_c_31_vec += a_3p_vec * b_p0_b_p1_vec;
                c_32_c_33_vec += a_3p_vec * b_p2_b_p3_vec;
                // 用完b就加
            }
            // 更新result_matrix
            // 更新result_matrix
            result_matrix[i * p + j] += c_00_c_01_vec[0];
            result_matrix[i * p + j + 1] += c_00_c_01_vec[1];
            result_matrix[i * p + j + 2] += c_02_c_03_vec[0];
            result_matrix[i * p + j + 3] += c_02_c_03_vec[1];

            result_matrix[(i + 1) * p + j] += c_10_c_11_vec[0];
            result_matrix[(i + 1) * p + j + 1] += c_10_c_11_vec[1];
            result_matrix[(i + 1) * p + j + 2] += c_12_c_13_vec[0];
            result_matrix[(i + 1) * p + j + 3] += c_12_c_13_vec[1];

            result_matrix[(i + 2) * p + j] += c_20_c_21_vec[0];
            result_matrix[(i + 2) * p + j + 1] += c_20_c_21_vec[1];
            result_matrix[(i + 2) * p + j + 2] += c_22_c_23_vec[0];
            result_matrix[(i + 2) * p + j + 3] += c_22_c_23_vec[1];

            result_matrix[(i + 3) * p + j] += c_30_c_31_vec[0];
            result_matrix[(i + 3) * p + j + 1] += c_30_c_31_vec[1];
            result_matrix[(i + 3) * p + j + 2] += c_32_c_33_vec[0];
            result_matrix[(i + 3) * p + j + 3] += c_32_c_33_vec[1];
        }
    }
}
void ijk_4_4_reg_regb_sse3_block_packab_new(double *matrxi1, double *matrix2, double *result_matrix, int n, int p)
{
    static double PackA[BLOCKSIZE * BLOCKSIZE];
    static double PackB[BLOCKSIZE * BLOCKSIZE];
    for (int i_o = 0; i_o < N; i_o += BLOCKSIZE)
    {
        for (int k_o = 0; k_o < M; k_o += BLOCKSIZE)
        {
            for (int j_o = 0; j_o < P; j_o += BLOCKSIZE)
            {
                do_block_packAB_new(&matrxi1[i_o * n + k_o], &matrix2[k_o * p + j_o], &result_matrix[i_o * p + j_o], i_o, j_o, k_o, n, p, j_o == 0, PackA, PackB);
            }
        }
    }
}
void *matrix_multiplication_thread(void *arg)
{
    ThreadParams *params = (ThreadParams *)arg;
    int start_row = params->start_row;
    int end_row = params->end_row;
    double *matrix1 = params->matrix1;
    double *matrix2 = params->matrix2;
    double *result_matrix = params->result_matrix;
    int n = params->n;
    int p = params->p;
    double *PackA = params->PackA;
    double *PackB = params->PackB;
    // 为每个线程分配空间
    for (int i_o = 0; i_o < (end_row - start_row); i_o += BLOCKSIZE)
    {
        for (int k_o = 0; k_o < M; k_o += BLOCKSIZEK)
        {
            for (int j_o = 0; j_o < P; j_o += BLOCKSIZE)
            {
                do_block_packAB_new(&matrix1[i_o * n + k_o], &matrix2[k_o * p + j_o], &result_matrix[i_o * p + j_o], i_o, j_o, k_o, n, p, j_o == 0, PackA, PackB);
            }
        }
    }
    pthread_exit(NULL);
}
void many_thread(double *matrxi1, double *matrix2, double *result_matrix, int n, int p)
{

    ThreadParams params[NUM_THREADS];     // 线程的参数
    pthread_t threads[NUM_THREADS];       // 线程的队列
    for (int t = 0; t < NUM_THREADS; t++) // 创建多少个线程
    {
        // 最后的end是480
        int start_row = (N / NUM_THREADS) * t;
        int end_row = (N / NUM_THREADS) * (t + 1);
        params[t].start_row = (N / NUM_THREADS) * t; // 这里是按行分的
        params[t].end_row = (N / NUM_THREADS) * (t + 1);
        params[t].matrix1 = &matrxi1[start_row * n]; // 都是大数组的指针
        params[t].matrix2 = matrix2;
        params[t].result_matrix = &result_matrix[start_row * p];
        params[t].n = n;
        params[t].p = p;
        params[t].PackA = &(PackA_g[t][0]);
        params[t].PackB = &(PackB_g[t][0]);
        pthread_create(&threads[t], NULL, matrix_multiplication_thread, &params[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++)
    {
        pthread_join(threads[t], NULL);
    }
}
// 实际矩阵乘法入口
void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P])
{
    if (N < 512 or M < 512 or P < 512)
    {
        ijk_1_4_reg(*matrix1, *matrix2, *result_matrix, M, P);
    }
    else
    {
        many_thread(*matrix1, *matrix2, *result_matrix, M, P);
    }
    // ikj_4_4_reg(*matrix1, *matrix2, *result_matrix, M, P);
}

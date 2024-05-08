#include "multiply.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>

// # define MAX_THREADS 30 // 定义最大线程

// typedef struct{
//     double (*matrix1)[M];
//     double (*matrix2)[P];
//     double (*result_matrix)[P];
//     int start_row;
//     int end_row;
//     int end_col;
//     int mid;
// } ThreadData;

// // 定义全局变量，以便能够灵活释放线程
// pthread_t threads[MAX_THREADS];
// ThreadData thread_data[MAX_THREADS];
// pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;   //
//

// TODO: you should implement your code in this file, we will only call `matrix_multiplication` to
// test your implementation

// 将第一个矩阵的每一行存入一个块中
// void block_matrix_row(double *matarix, double *block, int block_cols, int block_rows){
//     for(int i=0; i<block_rows; i++){
//         for(int j=0; j<block_cols; j++){

//         }
//     }
// }

// 函数用于执行矩阵乘法的每个分块的计算
void *matrix_multiply_block(void *arg)
{
    ThreadData *data = (ThreadData *)arg;
    int result_row = data->start_row;
    int result_end_row = data->end_row;

    // // 计算被分配的矩阵计算
    // for(int j=0; j<data->mid; j++){
    //     for(int i=0; i<data->end_col; i++){
    //         data->result_matrix[result_row][i] += data->matrix1[result_row][j] * data->matrix2[j][i];
    //     }
    // }

    // 进行计算
    for (int row = result_row; row < result_end_row; row++)
    {
        for (int mid = 0; mid < data->mid; mid++)
        {
            for (int col = 0; col < data->end_col; col++)
            {
                data->result_matrix[row][col] += data->matrix1[row][mid] * data->matrix2[mid][col];
            }
        }
    }

    // pthread_mutex_lock(&mutex);
    // thread_count--;
    // pthread_mutex_unlock(&mutex);

    return NULL;
}

void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P])
{
    // for(int row = 0; row < N; ++row)
    //     for(int mid = 0; mid < M; ++mid)
    //         for(int col = 0; col < P; ++col)
    //             result_matrix[row][col] += matrix1[row][mid] * matrix2[mid][col];
    // int block_rows = N;
    // int block_cols = P;

    // 为第一个矩阵分块

    // 为第二个矩阵分块

    //
    int MAX_THREADS = 16;
    // printf("%d", MAX_THREADS);
    // 定义全局变量，以便能够灵活释放线程
    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];

    // 计算
    int num_thread = 0; // 计算每个线程需要负责多少行
    num_thread = N / MAX_THREADS + 1;
    int thread_count = 0;

    for (int i = 1; i <= MAX_THREADS; i++)
    {
        // pthread_mutex_lock(&mutex);
        ThreadData *data = &thread_data[thread_count];

        data->matrix1 = matrix1;
        data->matrix2 = matrix2;
        data->result_matrix = result_matrix;
        data->start_row = (i - 1) * num_thread;
        data->end_row = i * num_thread;
        data->end_col = P;
        data->mid = M;

        if (data->end_row >= N)
        {
            data->end_row = N;
            i = MAX_THREADS + 1;
        }

        // 创建线程
        pthread_create(&threads[thread_count], NULL, matrix_multiply_block, (void *)data);

        thread_count++;

        // pthread_mutex_unlock(&mutex);
    }
    // 等待剩余线程完成
    for (int i = 0; i < MAX_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // // printf("矩阵1\n");
    // // print_matrix(matrix1);
    // // printf("矩阵2\n");
    // // print_matrix(matrix2);
    // printf("结果矩阵\n");
    // print_matrix(result_matrix);
}

// // 输出结果矩阵
// void print_matrix(double matrix[N][P]) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < P; j++) {
//             // 假设矩阵中的元素是双精度浮点数（double）
//             // 以整数形式输出
//             printf("%f ", matrix[i][j]);
//         }
//         printf("\n");
//     }
// }

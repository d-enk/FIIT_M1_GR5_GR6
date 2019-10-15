#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;

void read_B(vector<double> &vec, int n, int b, string st)
{
    ifstream in(st);
    vec.resize(n * (n + b) / 2);
    for (int i = 0; i < n / b; ++i)
        for (int bi = 0; bi < b; ++bi)
        {
            for (int j = 0, indst = 0; j <= i; ++j)
            {
                for (int bj = 0; bj < b; ++bj)
                    in >> vec[indst + (i - j) * b * b + bi * b + bj];

                indst += (n / b - j) * b * b;
            }
            for (int bj = 0; bj < n - b * (i + 1); ++bj)
                in >> st;
        }

    in.close();
}

void read_A(vector<double> &vec, int n, int b, string st)
{
    ifstream in(st);
    vec.resize(n * (n + b) / 2);
    for (int i = 0; i < n / b; ++i)
        for (int bi = 0; bi < b; ++bi)
        {
            for (int bj = 0; bj < b * i; ++bj)
                in >> st;

            for (int j = 0; j < n / b - i; ++j)
                for (int bj = 0; bj < b; ++bj)
                    in >> vec[(i + j) * b * b * (i + j + 1) / 2 + i * b * b + b * bi + bj];
        }

    in.close();
}

void mult(vector<double> &A, vector<double> &B, vector<double> &C, int n, int b)
{
    int blocksCount = n / b;
    int blockSize = b * b;

#pragma omp parallel for schedule(auto)
    for (int i = 0; i < blocksCount; ++i)
    {
        int ib = i * blockSize;
        int ai = ib * (i + 1) / 2; // Transponate str beginning
        int bi = 0;

        int ci = i * b * n;
        for (int j = 0; j < blocksCount; ++j)
        {
            int kt = j;
            for (; kt < i; ++kt)
            {
                for (int j = 0; j < b; ++j)
                    for (int i = 0; i < b; ++i)
                        for (int k = 0; k < b; ++k)
                            C[ci + j * b + i] += A[ai + kt * blockSize + k * b + j] * B[bi + k * b + i];
                bi += blockSize;
            }

            for (int k = kt; k < blocksCount; ++k)
            {
                int ai = k * blockSize * (k + 1) / 2 + ib;
                for (int j = 0; j < b; ++j)
                    for (int i = 0; i < b; ++i)
                        for (int k = 0; k < b; ++k)
                            C[ci + j * b + i] += A[ai + j * b + k] * B[bi + k * b + i];
                bi += blockSize;
            }

            ci += blockSize;
        }
    }
}

int main()
{

    int n = 2880;
    for (int b : {2, 4, 6, 8, 12, 16, 20, 24, 36, 48, 72, 2880})
    {
        vector<double> A;
        vector<double> B;
        read_A(A, n, b, "a.txt"); // simetric
        read_B(B, n, b, "b.txt"); // down treangualr
        vector<double> C(n * n);

        auto start = std::chrono::system_clock::now();
        mult(A, B, C, n, b);

        auto end = std::chrono::system_clock::now();

        int elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        cout << b << "\t"
             << elapsed_seconds << endl;
    }
    return 0;
}
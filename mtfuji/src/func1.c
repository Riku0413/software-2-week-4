#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "obj.h"

// パラメータ数の計算
int f_dimension()
{
    return 2;
}

// 誤差関数の値
double f_value(const double a[], const double x[], double y[], int N)
{
    double E = 0;
    for (int i = 0; i < N; i++) {
      E += (y[i] - a[0] * x[i] - a[1]) * (y[i] - a[0] * x[i] - a[1]);
    }
    return E;
}

// 勾配ベクトルの計算
void f_gradient(const double a[], const double x[], const double y[], double g[], int N)
{
    g[0] = 0;
    g[1] = 0;

    for (int i = 0; i < N; i++) {
      g[0] += -2 * x[i] * (y[i] - a[0] * x[i] - a[1]);
      g[1] += -2 * (y[i] - a[0] * x[i] - a[1]);
    }
}

// 勾配ノルムの計算 -> これがほぼゼロなら局所的な最小とみなせる
double calc_norm(const int dim, double v[])
{
    double tmp = 0;
    for (int i = 0; i < dim; i++) {
	tmp += v[i] * v[i];
    }
    tmp = sqrt(tmp);
    return tmp;
}

// データを渡して機械学習する関数
int optimize(const double alpha, const int dim, double a[], const double x[], const double y[], const int N, 
             void (*calc_grad)(const double [], const double [], const double [], double [], int))
{
    // 勾配ベクトルを記録する領域を確保
    double *g = malloc(dim * sizeof(double));
    
    int iter = 0;
    while (++iter < 10000) {
	
	// 引数で渡された関数を使って勾配ベクトルを計算
	(*calc_grad)(a, x, y, g, N);
	
	// 勾配ベクトルのノルムを評価
	const double norm = calc_norm(dim, g);
  if (norm < 0.01 | iter % 50 == 0) {
    printf("%3d norm = %7.4f", iter, norm);
    for (int i = 0; i < dim; i++) {
        printf(", a[%d] = %7.4f", i, a[i]);
    }
    printf("\n");
  }
	
	if (norm < 0.01) break;
	
	// 最急降下法による更新
	for (int i = 0; i < dim; i++) {
	    a[i] -= alpha * g[i];
	}
    }
    
    free(g);
    
    return iter;
}


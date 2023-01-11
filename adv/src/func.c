#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "object.h"

// パラメータ数の計算
int f_dimension()
{
    return 9; // 特徴量8, 定数項1, パラメータ9, 予測値1
}

// データの標準化
void normalize(const int N, const int dim, const double x[N][dim - 1], double norm_x[N][dim - 1], double sta[2][dim - 1])
{

  for (int k = 0; k < dim - 1; k++) {

    double ave = 0;
    for (int i = 0; i < N; i++) {
      ave += x[i][k];
    }
    ave /= N;

    double str = 0;
    for (int i = 0; i < N; i++) {
      str += (x[i][k] - ave) * (x[i][k] - ave);
    }
    str /= N;
    str = sqrt(str);

    for (int i = 0; i < N; i++) {
      norm_x[i][k] = (x[i][k] - ave) /str;
    }

    // 統計量を保存
    sta[0][k] = ave;
    sta[1][k] = str;
  }

}

// 平均平方二乗誤差 RMSE
double f_value(const int N, const int dim, const double a[], const double norm_x[N][dim - 1], const double y[])
{
    double E = 0;

    for (int i = 0; i < N; i++) {

      double y_pred = a[dim - 1];
      for (int k = 0; k < dim - 1; k++) {
        y_pred += a[k] * norm_x[i][k];
      }

      E += (y[i] - y_pred) * (y[i] - y_pred);
    }

    return sqrt(E / N);
}

// 勾配ベクトルの計算
void f_gradient(const int N, const int dim, const double a[], const double norm_x[N][dim - 1], const double y[], double g[])
{
    for (int k = 0; k < dim; k++) {
      g[k] = 0;
    }

    for (int i = 0; i < N; i++) {
      
      double y_pred = a[dim - 1];
      for (int k = 0; k < dim - 1; k++) {
        y_pred += a[k] * norm_x[i][k];
      }

      for (int k = 0; k < dim - 1; k++) {
        g[k] += -2 * norm_x[i][k] * (y[i] - y_pred);
      }

      g[dim - 1] += -2 * (y[i] - y_pred);
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

// 整形済みデータを渡して機械学習する関数
int optimize(const double alpha, const int N, const int dim, double a[], const double norm_x[N][dim - 1], const double y[], 
             void (*calc_grad)(const int, const int, const double [], const double [N][dim - 1], const double [], double []))
{
    // 勾配ベクトルを記録する領域を確保
    double *g = malloc(dim * sizeof(double));

    int iter = 0;
    while (++iter < 1000) {

	// 引数で渡された関数を使って勾配ベクトルを計算
	(*calc_grad)(N, dim, a, norm_x, y, g);

	// 勾配ベクトルのノルムを評価
	const double norm = calc_norm(dim, g);
  const double E = f_value(N, dim, a, norm_x, y);
  if (norm < 0.01 | iter % 10 == 0) {
    printf("%3d norm = %7.4f, RMSE = %7.4f", iter, norm, E);
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

// 整形前のデータから予測
double predict(const int dim, const double a[], const double data[], const double sta[2][dim - 1])
{
  double y_pred = a[dim - 1];

  double norm_data[8];

  // 標準化
  for (int k = 0; k < dim - 1; k++) {
    norm_data[k] = (data[k] - sta[0][k]) / sta[1][k];
  }

  for (int k = 0; k < dim - 1; k++) {
    y_pred += a[k] * norm_data[k];
  }

  return y_pred;
}

// データの分割
void divide(const int N, const int dim, const double rate, const double norm_x[N][dim - 1], double train_norm_x[N][dim - 1], double test_norm_x[N][dim - 1], const double y[], double train_y[], double test_y[])
{
  int train_n = N * rate;
  int test_n = N - train_n;

  for (int i = 0; i < train_n; i++) {
    for (int k = 0; k < dim - 1; k++) {
      train_norm_x[i][k] = norm_x[i][k];
    }
    train_y[i] = y[i];
  }
  for (int i = 0; i < test_n; i++) {
    for (int k = 0; k < dim - 1; k++) {
      test_norm_x[i][k] = norm_x[train_n + i][k];
    }
    test_y[i] = y[train_n + i];
  }

}
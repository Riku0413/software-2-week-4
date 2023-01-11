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

    // 訓練データの統計量を保存
    sta[0][k] = ave;
    sta[1][k] = str;
  }

}

// 平均二乗誤差 MSE
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

    return E / N;
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
    printf("%3d norm = %7.4f, MSE = %7.4f", iter, norm, E);
    // for (int i = 0; i < dim; i++) {
    //     printf(", a[%d] = %7.4f", i, a[i]);
    // }
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





int main(const int argc, const char **argv)
{
    // 引数の個数が1の時だけ、alpha に第1引数を採用し、それ以外は0.001
    const double alpha = (argc == 2) ? atof(argv[1]) : 0.001;
    
    const int dim = f_dimension();
    
    double *a = malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) {
      a[i] = 1/(i + 1);
    }
    
    printf("alpha = %f\n", alpha);
    
    double x[500][8];
    double norm_x[500][8];
    double y[500];
    double sta[2][8];

    // csvの読み込み
    Sample data[500];
    FILE *fp;
    if ((fp = fopen("housing.csv", "rb")) == NULL) {
      printf("csv cannot be read.\n");
      return 0;
    }

    char dummy[100];
    fscanf(fp, "%[^\n]", dummy);

    // データの読み込み
    for (int i = 0; i < 500; i++) {

      fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &data[i].number, &data[i].med, &data[i].age, &data[i].rooms, &data[i].bedrms, &data[i].pop, &data[i].occup, &data[i].latitude, &data[i].longtitude, &data[i].price);
      
      x[i][0] = data[i].med;
      x[i][1] = data[i].age;
      x[i][2] = data[i].rooms;
      x[i][3] = data[i].bedrms;
      x[i][4] = data[i].pop;
      x[i][5] = data[i].occup;
      x[i][6] = data[i].latitude;
      x[i][7] = data[i].longtitude;
      y[i] = data[i].price;

    }
    fclose(fp);

    // データの標準化
    for (int k = 0; k < dim - 1; k++) {
      normalize(500, f_dimension(), x, norm_x, sta);
    }

    // データの分割
    double train_norm_x[500][8];
    double test_norm_x[500][8];
    double train_y[500];
    double test_y[500];

    divide(500, dim, 0.8, norm_x, train_norm_x, test_norm_x, y, train_y, test_y);

    int train_n = 500 * 0.8;
    int test_n = 500 - train_n;

    // 学習
    const int iter = optimize(alpha, train_n, dim, a, train_norm_x, train_y, f_gradient);

    printf("number of iterations = %d\n", iter);

    double E_test = f_value(test_n, dim, a, test_norm_x, test_y);
    printf("MSE of test data = %7.4f\n", E_test);

    for (int i = 0; i < 10; i++) {
      double y_pred = predict(dim, a, x[i], sta);
      printf("y_pred = %lf, y = %lf\n", y_pred, y[i]);
    }

    free(a);

    return 0;
}

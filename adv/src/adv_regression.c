#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "object.h"

int main(const int argc, const char **argv)
{
    // 引数の個数が1の時だけ、alpha に第1引数を採用し、それ以外は0.001
    const double alpha = (argc == 2) ? atof(argv[1]) : 0.001;
    
    const int dim = f_dimension();

    // パラメータの初期化
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
    printf("\n- 学習 -\n");
    const int iter = optimize(alpha, train_n, dim, a, train_norm_x, train_y, f_gradient);
    printf("number of iterations = %d\n", iter);

    // 最終的なパラメータの表示
    printf("\n- 学習後のパラメータ -\n");
    for (int k = 0; k < dim; k++) {
      printf("a[%d] = %7.4f", k, a[k]);
      if (k < dim - 1) {
        printf(", ");
      }
    }
    printf("\n");

    // テスト
    double E_test = f_value(test_n, dim, a, test_norm_x, test_y);
    printf("\n- テスト -\n");
    printf("RMSE of test data = %7.4f\n", E_test);

    printf("\n- テストデータを用いた予測 -\n");
    for (int i = 0; i < 5; i++) {
      double y_pred = predict(dim, a, x[i], sta);
      printf("y_pred[%d] = %lf, y[%d] = %lf\n", i, y_pred, i, y[i]);
    }

    free(a);

    return 0;
}


// 構造体の定義
typedef struct {
    char loc[20];   // location name
    double alt;  // altitude (km)
    double temp; // temperature (centigrade)
} Sample;

// パラメータ数の計算
int f_dimension();
// 誤差関数の値
double f_value(const double a[], const double x[], double y[], int N);
// 勾配ベクトルの計算
void f_gradient(const double a[], const double x[], const double y[], double g[], int N);
// 勾配ノルムの計算 -> これがほぼゼロなら局所的な最小とみなせる
double calc_norm(const int dim, double v[]);
// データを渡して機械学習する関数
int optimize(const double alpha, const int dim, double a[], const double x[], const double y[], const int N, 
             void (*calc_grad)(const double [], const double [], const double [], double [], int));

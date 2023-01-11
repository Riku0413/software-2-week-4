// 構造体の定義
typedef struct {
    double number; //
    double med; //
    double age; //
    double rooms; //
    double bedrms; //
    double pop; //
    double occup; //
    double latitude; //
    double longtitude; //
    double price; //
} Sample;

// パラメータ数の計算
int f_dimension();
// データの標準化
void normalize(const int N, const int dim, const double x[N][dim - 1], double norm_x[N][dim - 1], double sta[2][dim - 1]);
// 平均二乗誤差 MSE
double f_value(const int N, const int dim, const double a[], const double norm_x[N][dim - 1], const double y[]);
// 勾配ベクトルの計算
void f_gradient(const int N, const int dim, const double a[], const double norm_x[N][dim - 1], const double y[], double g[]);
// 勾配ノルムの計算 -> これがほぼゼロなら局所的な最小とみなせる
double calc_norm(const int dim, double v[]);
// 整形済みデータを渡して機械学習する関数
int optimize(const double alpha, const int N, const int dim, double a[], const double norm_x[N][dim - 1], const double y[], 
             void (*calc_grad)(const int, const int, const double [], const double [N][dim - 1], const double [], double []));
// 整形前のデータから予測
double predict(const int dim, const double a[], const double data[], const double sta[2][dim - 1]);
// データの分割
void divide(const int N, const int dim, const double rate, const double norm_x[N][dim - 1], double train_norm_x[N][dim - 1], double test_norm_x[N][dim - 1], const double y[], double train_y[], double test_y[]);

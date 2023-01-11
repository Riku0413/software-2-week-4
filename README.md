# software-2-week-4

## 課題1

### 実行方法

まず,以下のコードによりディレクトリを移動する.
```
cd ./paint
```

続いて,以下のmakeコマンドにより,コンパイルする.
```
make
```

最後に,例えば以下のコードによって実行する.
```
./bin/paintapp 80 40
```

## 課題2

### プログラムの要点

1. 誤差関数や勾配の計算

誤差関数については以下の関数で計算.
```
double f_value(const double a[], const double x[], double y[], int N)
{
    double E = 0;
    for (int i = 0; i < N; i++) {
      E += (y[i] - a[0] * x[i] - a[1]) * (y[i] - a[0] * x[i] - a[1]);
    }
    return E;
}
```

それに合わせて,勾配ベクトルは以下の関数で計算.
```
void f_gradient(const double a[], const double x[], const double y[], double g[], int N)
{
    g[0] = 0;
    g[1] = 0;

    for (int i = 0; i < N; i++) {
      g[0] += -2 * x[i] * (y[i] - a[0] * x[i] - a[1]);
      g[1] += -2 * (y[i] - a[0] * x[i] - a[1]);
    }
}
```

2. csvファイルの読み込み

以下のコードによって,Sample構造体の配列を作り,csvファイルのデータを読みこむ.
```
double x[14] = {};
double y[14] = {};

Sample data[14];
FILE *fp;
if ((fp = fopen("data.csv", "rb")) == NULL) {
  printf("csv cannot be read.\n");
  return 0;
}
for (int i = 0; i < 14; i++) {

  fscanf(fp, "%[^,], %lf, %lf", data[i].loc, &data[i].alt, &data[i].temp);
  fgetc(fp); // '\n' を除去

  x[i] = data[i].alt;
  y[i] = data[i].temp;

}
fclose(fp);
```

3. データを標高の降順にソート

データのソートは,まず第1データをセットし,次に第2データを第1データと大小比較してランキング化する.
さらに,第３データを既存の2つと比較してランキングを更新する.これを繰り返し,完成した14データのランキングをもとに最後にまとめてソートする, というアルゴリズムで実装している.
```
// ソート
Sample data_sorted[14];
int rank[14]; // 0位, 1位, 2位 ...
rank[0] = 0; // 0位に　data 0 を代入

for (int i = 1; i < 14; i++) {
  for (int j = 0; j < i; j++) {
    if (data[i].alt > data[rank[j]].alt) {
      for (int k = 13; k > j; k--) {
        rank[k] = rank[k-1];
      }
      rank[j] = i;
      break;
    }
    if (j == i - 1) {
      rank[j + 1] = i;
    }
  }
}

for (int i = 0; i < 14; i++) {
  data_sorted[i] = data[rank[i]];
}

// ソートの確認
for (int i = 0; i < 14; i++) {
  printf("%s, %lf, %lf\n", data_sorted[i].loc, data_sorted[i].alt, data_sorted[i].temp);
}
```

### 実行方法

まず,以下のコードによりディレクトリを移動する.
```
cd ./mtfuji
```

続いて,以下のmakeコマンドにより,コンパイルする.
```
make
```

最後に,例えば以下のコードによって実行する.
```
./bin/mtfuji
```


## 課題3

### 使用したデータについて

以下のURLからアクセス, ダウンロードできる.

http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

カリフォルニアの住宅価格を８種類の説明変数から予測するデータセットである.

### プログラムのフロー

0. pythonでオープンデータを読み込み,csv化

　　(このとき使用したpyファイル,作成したcsvファイルは同封済み)

1. csvファイルの読み込み
2. パラメータの初期化
3. データの標準化
4. データの分割
5. 訓練データによるモデルの訓練
6. テストデータによる性能評価

### プログラムのポイント

・データの標準化

効率的に学習を進めるために,以下の関数によって全特徴量を標準化できるようにした.
```
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
```

・データの分割

モデルの汎化性能および過学習の傾向を調べるために,以下の関数でデータの分割を可能にした.
rateは訓練データの割合を表す.
```
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
```

・テストデータによる性能評価

以下の関数により,平均平方二乗誤差(RMSE)を計算できるようにし,モデルの汎化性能および過学習の傾向を調べられるようにした.
```
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
```

### 実行方法

まず,以下のコードによりディレクトリを移動する.
```
cd ./adv
```

続いて,以下のmakeコマンドにより,コンパイルする.
```
make
```

最後に,例えば以下のコードによって実行する.
```
./bin/adv_regression
```
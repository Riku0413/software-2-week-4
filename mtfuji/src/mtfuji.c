#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "obj.h"

int main(const int argc, const char **argv)
{
    // 引数の個数が1の時だけ、alpha に第1引数を採用し、それ以外は0.01
    const double alpha = (argc == 2) ? atof(argv[1]) : 0.01;
    
    const int dim = f_dimension();
    
    double *a = malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) {
	a[i] = 0;
    }
    
    printf("alpha = %f\n", alpha);
    
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

    const int iter = optimize(alpha, dim, a, x, y, 14, f_gradient);
    
    printf("number of iterations = %d\n", iter);
    
    free(a);
    
    double y_pred = a[0] * 3.776 + a[1];
    printf("y_pred = %lf\n", y_pred);
    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <sys/time.h>
#include "../ulmblas.h"
#define M 1024
#define K 128
#define N 1024

void
ULMBLAS(dgemm)(const enum Trans  transA,
               const enum Trans  transB,
               const int         m,
               const int         n,
               const int         k,
               const double      alpha,
               const double      *A,
               const int         ldA,
               const double      *B,
               const int         ldB,
               const double      beta,
               double            *C,
               const int         ldC);
int main(){
  // init
  double (*a)[K] = (double (*)[K])malloc(sizeof(double)*M*K);
  double (*b)[N] = (double (*)[N])malloc(sizeof(double)*N*K);
  double (*c)[N] = (double (*)[N])malloc(sizeof(double)*M*N);
  struct timeval stop, start;
  for(int i=0;i<M;i++){
    for(int j=0;j<K;j++){
      a[i][j] = 0.33*(float)i+0.29*(float)j;
    }
  }
  for(int i=0;i<K;i++){
    for(int j=0;j<N;j++){
      b[i][j] = 0.67*(float)i+0.98*(float)j;
    }
  }
  //C=A*B^T
  // openblas
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      c[i][j] = 0;
    }
  }
  gettimeofday(&start, NULL);
  cblas_dgemm(CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              M,N,K,1,(double*)a,K,(double*)b,N,0,(double*)c,N);
  gettimeofday(&stop, NULL);
  printf("li took %lu get %lf\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec, c[1][1]);
  // me
  /*
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      c[i][j] = 0;
    }
  }
  gettimeofday(&start, NULL);
  for(int i=0;i<M;i++)
    for(int j=0;j<N;j++){
      double *cp = &c[i][j];
      double *ap = a[i];
      double *bp = b[j];
      for(int p=0;p<K;p++){
        *cp += *(ap++) * *(bp++);
      }
    }
  gettimeofday(&stop, NULL);
  printf("me took %lu get %lf\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec, c[1][1]);
  */
  // ulm
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      c[i][j] = 0;
    }
  }
  gettimeofday(&start, NULL);
  ULMBLAS(dgemm)(Trans,
         Trans,
         M,N,K,1,(double*)a,K,(double*)b,N,0,(double*)c,N);
  gettimeofday(&stop, NULL);
  printf("ul took %lu get %lf\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec, c[1][1]);
  return 0;
}

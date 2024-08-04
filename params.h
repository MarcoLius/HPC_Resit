#pragma once
#include <math.h>

const int N1 = 32;
const int N2 = 32;
const int N3 = 32;
const int M = 50000;
const int mm= 1;
const int l = 3; // l >= 3
const int ml = (l - 1) / 2;
const double r = 3.81;

void init(double u[N1][N2][N3]);

void step(double u[N1][N2][N3], const double du[N1][N2][N3]);

void dudt(const double u[N1][N2][N3], double du[N1][N2][N3]);

void stat(double* stats, const double u[N1][N2][N3]);

/*
  Define simple integer min/max functions
*/
int imin(int a, int b){
  if (a <= b){
    return a;
  }else{
    return b;
  }
}

int imax(int a, int b){
  if (a <= b){
    return b;
  }else{
    return a;
  }
}

/*
  Define simple deterministic initialization function
*/
double u0(int n1, int n2, int n3){
    return (0.2*n1)/N1 + (0.3*n2)/N2 + (0.5*n3)/N3;
}


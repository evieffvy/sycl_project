#pragma once
// Regularized incomplete gamma functions used by the Block Frequency Test.
// These are not provided by <cmath> and must be implemented explicitly.
//
//   igam (a, x) = P(a, x) = lower  regularized incomplete gamma
//   igamc(a, x) = Q(a, x) = upper  regularized incomplete gamma  (= 1 - P)

double igam (double a, double x);
double igamc(double a, double x);

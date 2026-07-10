"""Active return attribution toolkit.

Two complementary explanations of a portfolio's active return vs. a benchmark:

* Module A (``attribution.brinson``): holdings-based Brinson-Fachler attribution
  with Carino geometric multi-period linking.
* Module B (``attribution.factor_model``): returns-based factor attribution
  (CAPM / FF3 / FF5 / Carhart) with hand-rolled OLS and Newey-West standard errors.

All attribution and regression math is implemented from first principles using
numpy only. ``statsmodels`` appears solely in the test suite as an independent
cross-check.
"""

__version__ = "0.1.0"

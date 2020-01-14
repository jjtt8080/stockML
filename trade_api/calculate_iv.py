import numpy as np
import scipy.stats  as si



def newton_vol_call(S, K, T, C, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # C: Call value
    # r: interest rate
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - C) / vega

    return abs(xnew)


def newton_vol_put(S, K, T, P, r, sigma):
    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - P) / vega

    return abs(xnew)

def newton_vol_call_div(S, K, T, C, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S   / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    fx = S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.exp(-q * T) * np.sqrt(T) * np.exp((-si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - C) / vega

    return abs(xnew)

def newton_vol_put_div(S, K, T, P, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0) - P

    vega = (1 / np.sqrt(2 * np.pi)) * S * np.exp(-q * T) * np.sqrt(T) * np.exp((-si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1
    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew -  fx - P) / vega


    return abs(xnew)
#print(newton_vol_call(25, 20, 1, 7, 0.05, 0.25))
#print(newton_vol_put(25, 20, 1, 7, 0.05, 0.25))

#print(newton_vol_call_div(58.84, 55, 8, 3.90, 0.0417, 0.030, 0.1385))
#print(newton_vol_put_div(58.84, 52, 1, 0.01, 0.0417, 0.030, 0.1385))
#print(newton_vol_put_div(58.84, 53, 1, 0.01, 0.0417, 0.030, 0.1385))
#print(newton_vol_put_div(58.84, 55, 1, 0.01, 0.0417, 0.030, 0.1385))
#print(newton_vol_put_div(58.84, 56, 1, 0.03, 0.0417, 0.030, 0.1385))
#print(newton_vol_put_div(58.84, 58, 7, 0.12, 0.0417, 0.030, 0.1385))

#print(newton_vol_call_div(58.84, 54, 8, 4.95, 0.0417, 0.030, 0.1385))


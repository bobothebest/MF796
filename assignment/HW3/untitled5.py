# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:56:40 2024

@author: ASUS
"""
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import interpolate
'''
# (a) Use ppf to inverse delta, which help us getting K
def cal_strikes(df,s0=100,r=0):
    sigma1 = df.iloc[:,1]
    sigma2 = df.iloc[:,2]
    m1k = []
    m3k = []
    for i  in range(len(df.iloc[:,0])):
        if df.iloc[i,3] == 'call':
            d1 = norm.ppf(df.iloc[i,0])
            k1 = s0 * np.exp(0.5 * sigma1[i] ** 2 * 1/12 - sigma1[i] * np.sqrt(1/12) * d1)
            k3 = s0 * np.exp(0.5 * sigma2[i] ** 2 * 3/12 - sigma2[i] * np.sqrt(3/12) * d1)
            m1k +=[k1]
            m3k +=[k3]
        else:
            d1 = norm.ppf(df.iloc[i,0])
            k1 = s0 * np.exp(0.5 * sigma1[i] ** 2 * 1/12 + sigma1[i] * np.sqrt(1/12) * d1)
            k3 = s0 * np.exp(0.5 * sigma2[i] ** 2 * 3/12 + sigma2[i] * np.sqrt(3/12) * d1)
            m1k += [k1]
            m3k +=[k3]
    df['1M_Strike'] = m1k
    df['3M_Strike'] = m3k
    return df
m1 = [0.3225,0.2473,0.2012,0.1824,0.1574,0.1370,0.1148]
m3 = [0.2836,0.2178,0.1818,0.1645,0.1462,0.1256,0.1094]
delta = [0.1,0.25,0.4,0.5,0.4,0.25,0.1]
pc = ['put','put','put','call','call','call','call']
df = pd.DataFrame({'Delta':delta,'1M':m1,'3M':m3,'Put or Call':pc})
df = cal_strikes(df)

# (b)
def interpolate(df):
    sigma1 = df.iloc[:,1]
    sigma3 = df.iloc[:,2]
    k1 = df.iloc[:,4]
    k3 = df.iloc[:,5]
    f1 = np.polyfit(k1,sigma1,3)
    f2 = np.polyfit(k3,sigma3,3)
    n = np.arange(84,108,0.01)
    plt.figure(num=3,figsize=(8,5))
    plt.plot(n,np.polyval(f1,n),label='1M')
    plt.plot(n,np.polyval(f2,n),label='3M')
    plt.legend()
    plt.xlabel('Strike Price')
    plt.ylabel('Volatility')
    plt.show() 
    return f1, f2
sigma1,sigma2=interpolate(df)
# (c)
def blackscholes(s0,k,t,r,sigma):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    price = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    return price

def density(sigma1,sigma2,s0=100,r=0):
    t1 = 1/12
    t2 = 3/12
    h = 0.01
    k = list(np.arange(65,120,0.5))
    d1_list = []
    d2_list = []
    for i in range(len(k)):
        d1 = blackscholes(s0,k[i]-h,t1,r,max(0.00001,np.polyval(sigma1,k[i]))) - 2 * blackscholes(s0,k[i],t1,r,max(0.00001,np.polyval(sigma1,k[i]))) + blackscholes(s0,k[i]+h,t1,r,max(0.00001,np.polyval(sigma1,k[i])))
        d1_list += [d1/h**2]
        d2 = blackscholes(s0,k[i]-h,t2,r,np.polyval(sigma2,k[i])) - 2 * blackscholes(s0,k[i],t2,r,np.polyval(sigma2,k[i])) + blackscholes(s0,k[i]+h,t2,r,np.polyval(sigma2,k[i]))
        d2_list += [d2/h**2] 
    plt.figure(num=3,figsize=(8,5))
    plt.plot(k,d1_list,label='1M')
    plt.plot(k,d2_list,label='3M')
    plt.legend()
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.show()         
    return d1_list,d2_list
d1_list,d2_list=density(sigma1, sigma2)    
# (d)
def density_50D(sigma1,sigma2,s0=100,r=0):
    t1 = 1/12
    t2 = 3/12
    h = 0.1
    k= list(np.arange(65,120,0.5))
    d1_list = []
    d2_list = []
    for i in range(len(k)):
        d1= blackscholes(s0,k[i]-h,t1,r,sigma1) - 2 * blackscholes(s0,k[i],t1,r,sigma1) + blackscholes(s0,k[i]+h,t1,r,sigma1)
        d1_list += [d1/h**2]
        d2 = blackscholes(s0,k[i]-h,t2,r,sigma2) - 2 * blackscholes(s0,k[i],t2,r,sigma2) + blackscholes(s0,k[i]+h,t2,r,sigma2)
        d2_list += [d2/h**2] 
    plt.figure(num=3,figsize=(8,5))
    plt.plot(k,d1_list,label='1M')
    plt.plot(k,d2_list,label='3M')
    plt.legend()
    plt.xlabel('Constant Delta Strike Price')
    plt.ylabel('Density')
    plt.show()         
    return d1_list,d2_list   
density_50D(0.1824,0.1645,s0=100,r=0)

# (e)
def digital_1M_110P(d1):
    p = 0
    for i in range(90):
        p += d1[i] * 0.5
    return p
digital_1M_110P(d1_list)

def digital_3M_105C(d2):
    c = 0
    for i in range(81,110):
        c += d2[i] * 0.5
    return c
digital_3M_105C(d2_list)

def call_2M_100(d1,d2):
    c = 0
    p = 0
    k = list(np.arange(65,120,0.5))
    for i in range(70):
        p += d1[i] * 0.5 * (100-k[i])
    for i in range(71,110):
        c += d2[i] * 0.5 * (k[i]-100)
    return (p+c)/2
call_2M_100(d1_list, d2_list)
'''
# Problem 2
# cleaning data
def clean_data(name):
    data = pd.read_excel(name)
    data['call_mid'] = (data.call_bid + data.call_ask) / 2
    data['put_mid'] = (data.put_bid + data.put_ask) / 2
    data_call = data[['expDays', 'expT', 'K','call_mid', 'call_ask', 'call_bid']]
    data_put = data[['expDays', 'expT', 'K', 'put_mid', 'put_ask', 'put_bid']]
    data
    return data_call, data_put, data
name='C:/Users/ASUS/Desktop/mf796-hw3-opt-data.xlsx'
call,put,df=clean_data(name)
call.describe()
# (a)
def check_mono(df, opt_type):
    mid_col = df.columns[df.columns.str.contains(
        'mid')][0]  # find the column containing 'mid'
    if opt_type == 'c':
        return any(df[mid_col].pct_change().dropna() >= 0)
    else:
        return any(df[mid_col].pct_change().dropna() <= 0)
def check_delta(df, opt_type):
    mid_col = df.columns[df.columns.str.contains('mid')][0]
    df['delta'] = (df[mid_col] - df[mid_col].shift(1)) / (df.K - df.K.shift(1))
    if opt_type == 'c':
        return any(df.delta >= 0) or any(df.delta < -1)
    else:
        return any(df.delta > 1) or any(df.delta <= 0)
def check_convex(df):
    mid_col = df.columns[df.columns.str.contains('mid')][0]
    df['convex'] = df[mid_col] - 2 * \
        df[mid_col].shift(1) + df[mid_col].shift(2)
    return any(df.convex < 0)
def check_arb(df, opt_type):
    r1 = check_mono(df, opt_type)
    r2 = check_delta(df, opt_type)
    r3 = check_convex(df)
    return pd.Series([r1, r2, r3], index=['Monotonic', 'Delta', 'Convexity'])   
c_clean= call.groupby('expDays').apply(check_arb, opt_type='c')
put_clean =put.groupby('expDays').apply(check_arb, opt_type='p')
c_clean,put_clean
# From here we can see there is no arbitrage.

# (b)
def heston_characteristic_eqn(u, sigma, k,p,s_0,r,t,theta, v_0):
    '''characteristic function of heston model'''
    lambd = np.sqrt((sigma**2)*((u**2)+1j*u) + (k - 1j*p*sigma*u)**2) 
    omega_numerator = np.exp(1j*u*np.log(s_0)+1j*u*(r-0.0177)*t+(1/(sigma**2))*k*theta*t*(k - 1j*p*sigma*u))
    omega_denominator = (np.cosh(0.5*lambd*t) + (1/lambd)*(k - 1j*p*sigma*u)*np.sinh(0.5*lambd*t))**((2*k*theta)/(sigma**2))
    phi = (omega_numerator/omega_denominator) * np.exp(-((u**2 + 1j*u)*v_0)/(lambd*(1/np.tanh(0.5*lambd*t)) + (k - 1j*p*sigma*u)))
    return phi

def calc_fft_heston_call_prices(alpha, N, delta_k, sigma, k, p, s_0, r, t, theta, v_0, K ):
    '''calculate call option price via FFT'''
    delta = np.zeros(N)
    delta[0] = 1 
    delta_v=2*np.pi/(N*delta_k)
    beta = np.log(K) - delta_k*N*0.5
    k_list = np.array([(beta +(i-1)*delta_k) for i in range(1,N+1) ])
    v_list = np.arange(N) * delta_v
    x_numerator = np.array( [((2-delta[i])*delta_v)*np.exp(-r*t)  for i in range(N)] )
    x_denominator = np.array( [2 * (alpha + 1j*i) * (alpha + 1j*i + 1) for i in v_list] )
    x_exp = np.array( [np.exp(-1j*(beta)*i) for i in v_list] )
    x_list = (x_numerator/x_denominator)*x_exp* np.array([heston_characteristic_eqn(i - 1j*(alpha+1),sigma, k,p,s_0,r,t,theta, v_0) for i in v_list])
    y_list = np.fft.fft(x_list)
    prices = np.array( [(1/np.pi) * np.exp(-alpha*(beta +(i-1)*delta_k)) * np.real(y_list[i-1]) for i in range(1,N+1)] )
    return prices,np.exp(k_list)


def call_mse(params,call,put,alpha=1.5,N=2**12,delta_k=0.01,s_0=267.15,r=0.015):
    k, theta, sigma, p, v_0 = params
    r1 = 0
    for t in call['expT'].unique():
        d = call[call.expT==t]
        K = np.array(d['K'])
        P= np.array(d['call_mid'])
        prices ,k_list= calc_fft_heston_call_prices(alpha, N, delta_k, sigma, k, p, s_0, r, t, theta, v_0, s_0)
        p_clean,k_clean=[],[]
        for i in range(len(k_list)):
            if( k_list[i]>10 )&(k_list[i] < 500):
                p_clean+=[prices[i]]
                k_clean+=[k_list[i]]
        tck = interpolate.splrep( k_clean,p_clean)
        price_K = interpolate.splev(K, tck).real
        r1 += np.sum((price_K - P)**2 )
    r2=0
    for t in put['expT'].unique():
        d = put[put.expT==t]
        K = np.array(d['K'])
        P= np.array(d['put_mid'])
        prices ,k_list= calc_fft_heston_call_prices(-alpha, N, delta_k, sigma, k, p, s_0, r, t, theta, v_0, s_0)
        p_clean,k_clean=[],[]
        for i in range(len(k_list)):
            if( k_list[i]>10 )&(k_list[i] < 500):
                p_clean+=[prices[i]]
                k_clean+=[k_list[i]]
        tck = interpolate.splrep( k_clean,p_clean)
        price_K = interpolate.splev(K, tck).real
        r2 += np.sum((price_K - P)**2 )
    return r1+r2

def callback1(params):
    global times
    if times % 5 == 0:
        print('{}: {}'.format(times, call_mse(params,call,put)))
    times += 1
'''
alpha=1
N=2**10
delta_k=0.3
s_0=267.15
K=240
r=0.015
t=49/365
k,theta,sigma,p,v_0=[0, 0.2, 0.2, 0, 0.2]
calc_fft_heston_call_prices(alpha, N, delta_k, sigma, k, p, s_0, r, t, theta, v_0, K)  
     
'''      
#params=[1.269,0.088,1,-0.8237,0.034]
#call_mse(params, call, put)
times=1
x0 = [1, 0.1, 0.5,  -0.5, 0.1]
lower = [0.01, 0.01, 0.0, -1, 0]
upper = [2.5, 1, 0.7, 0.5, 0.5]
bounds = tuple(zip(lower, upper))
opt1 = minimize(call_mse, x0, args=(call,put), method='Nelder-Mead', bounds=bounds, callback=callback1,options={'maxiter': 300})
print('The result is :',opt1.x,' and the square sum is about: ' , opt1.fun)

# Changing starting point
times=1
x0 = [0.2, 0.3, 0.3, 0.1, 0.3]
opt2 = minimize(call_mse, x0, args=(call,put), method='Nelder-Mead', bounds=bounds, callback=callback1,options={'maxiter': 300})
print('The result is :',opt2.x,' and the square sum is about: ' , opt2.fun)

# Changing the bound
times=1
lower = [0.01, 0.01, 0.0, -1, 0]
upper = [2.5, 1, 1, 0.5, 0.5]
x0 = [1, 0.1, 0.5,  -0.5, 0.1]
bounds = tuple(zip(lower, upper))
opt3 = minimize(call_mse, x0, args=(call,put), method='Nelder-Mead', bounds=bounds, callback=callback1,options={'maxiter': 300})
print('The result is :',opt3.x,' and the square sum is about: ' , opt3.fun)
# Problem 3
def bsmodel(K, s0, sigma, r, q, T):
    d1 = (np.log(s0/K)+(r-q+sigma**2/2)*T)/(sigma*T**0.5)
    d2 = d1- sigma*T**0.5
    return(norm.cdf(d1)*s0-norm.cdf(d2)*K*np.exp(-r*T))

def bsvol(price,k,s0,r,t,q):
    def object_fun(vol):
        return bsmodel(k, s0, vol, r, q, t)-price
    vol=fsolve(object_fun,0.5)[0]
    return vol
alpha = 1
N=2**10
delta_k=0.01
s_0 = 267.15
q = 0.0177
r = 0.015
t = 3/12
K = 275
k = 1.2663783
theta = 0.08852744
sigma = 0.99983564
p = -0.8235954 
v_0 = 0.03430315
price=calc_fft_heston_call_prices(alpha, N, delta_k, sigma, k, p, s_0, r, t, theta, v_0, K)[0][N//2]
vol_bs=bsvol(price, K, s_0, r, t,q)
def greekbs(k,s0,vol,r,q,t):
    d1 = (np.log(s0/k)+(r-q+vol**2/2)*t)/(vol*(t)**0.5)
    delta = np.exp(-q*t)*norm.cdf(d1)
    vega = np.exp(-q*t)*s0*(t**0.5)*norm.cdf(d1)
    return delta,vega,d1
delta_bs,vega_bs,d1_bs=greekbs(K, s_0, vol_bs, r, q, t)

delta_heston=(calc_fft_heston_call_prices(alpha, N, delta_k, sigma, k, p, s_0+0.01, r, t, theta, v_0, K)[0][N//2]-calc_fft_heston_call_prices(alpha, N, delta_k, sigma, k, p, s_0-0.01, r, t, theta, v_0, K)[0][N//2])/0.02
dv = 0.01 * v_0
vega_heston=(calc_fft_heston_call_prices(alpha, N, delta_k, sigma, k, p, s_0, r, t, theta+dv, v_0+dv, K)[0][N//2]-calc_fft_heston_call_prices(alpha, N, delta_k, sigma, k, p, s_0-0.01, r, t, theta-dv, v_0-dv, K)[0][N//2])/2/dv









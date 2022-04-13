#Fanghui Shao
import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from scipy import interpolate
import warnings
#(a) Calibirate Heston Model
class Heston():
    def __init__(self, params, T, S0, r, q):
        self.S0 = S0
        self.T = T
        self.r = r
        self.q = q
        self.sigma = params[0]
        self.v0 = params[1]
        self.kappa = params[2]
        self.rho = params[3]
        self.theta = params[4]
        
    def lambda_func(self, u):
        return np.sqrt(self.sigma**2*(u**2+u*1j) + (self.kappa - self.rho*self.sigma*u*1j)**2)

    def omega_func(self, u):
        S0 = self.S0
        r = self.r
        q = self.q
        T = self.T
        sigma = self.sigma
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        lamb = self.lambda_func(u)
        num = np.exp(u*1j * np.log(S0) + u*(r-q)*T*1j + (kappa*theta*T*(kappa-rho*sigma*u*1j))/sigma**2)
        denom = (np.cosh(lamb*T/2) + (kappa-rho*sigma*u*1j)/lamb * np.sinh(lamb*T/2)) ** (2*kappa*theta/sigma**2)
        return num / denom
    
    def phi_func(self, u):
        lamb = self.lambda_func(u)
        omega = self.omega_func(u)
        num = -(u**2+u*1j)*self.v0
        denom = lamb/np.tanh(lamb*self.T/2) + self.kappa - self.rho*self.sigma*u*1j
        return omega * np.exp(num / denom)
    
    def FFT_price(self, K, N, B, alpha):
        r = self.r
        T = self.T
        S0 = self.S0
        dv = B / N
        dk = 2 * np.pi / B
        v_j = (np.arange(1,N+1) - 1)*dv
        delta_j = np.zeros(N)
        delta_j[0] = 1
        x_j = (2-delta_j)*dv * np.exp(-r*T) / (2*(alpha+v_j*1j)*(alpha+v_j*1j+1)) *np.exp(-1j*(np.log(S0)-dk*N/2)*v_j)*self.phi_func(v_j - (alpha + 1)*1j)
        y_j = np.fft.fft(x_j)
        call_j = y_j.real * np.exp(-alpha*(np.log(S0)-dk*(N/2-np.arange(N)))) / np.pi
        
        beta = np.log(S0) - dk * N /2
        k_j = beta + (np.arange(1,N+1) - 1)*dk
        curve = interpolate.splrep(k_j, call_j)
        k = np.log(K)
        fft_price = interpolate.splev(k, curve)
        return fft_price
     
def FFT_price(params, K, N, B, alpha, T, params2):
    price = Heston(params,T, params2[0], params2[1], params2[2]).FFT_price(K, N, B, alpha)
    return price

def mse_fun(df, params, N, B, alpha, params2):
    res = 0
    for T in df.expT.unique():
        temp = df[df.expT == T]
        K = temp.K
        prices = FFT_price(params, K, N, B, alpha, T, params2)
        res += sum((prices - temp.mid_price) ** 2)
    return res

def target_fun(params, df_call, df_put, N, B, alpha, params2):
    res = 0
    res += mse_fun(df_call, params, N, B, alpha, params2)
    res += mse_fun(df_put, params, N, B, -alpha, params2)
    return res


warnings.filterwarnings('ignore')
df = pd.read_excel('mf796-hw3-opt-data.xlsx')
df_call = df[['expDays', 'expT', 'K', 'call_bid', 'call_ask']]
df_put = df[['expDays', 'expT', 'K', 'put_bid', 'put_ask']]
df_call['mid_price'] = (df_call['call_ask'] + df_call['call_bid']) / 2
df_put['mid_price'] = (df_put['put_ask'] + df_put['put_bid']) / 2
df_call['spread'] = df_call['call_ask'] - df_call['call_bid']
df_put['spread'] = df_put['put_ask'] - df_put['put_bid']

S0 = 267.15
r = 0.015
q = 0.0177
params2 = [S0, r, q]
B = 1000
N = 2**15
dv = B / N
alpha = 1.5

args = (df_call, df_put, N, B, alpha, params2)
x1 = [0.7, 0.2, 1, -1, 0.5]
bnds1 = ((0, 2), (0, 1), (0.1, 5), (-1, 1), (0, 2))
print('Start Calibration:')
params_star = minimize(target_fun, np.array(x1), args=args, method='SLSQP', bounds=bnds1)

print('Calibration Success?',params_star.success)
print('Calibration Result:',params_star.x)


class FFT:
    def __init__(self, lst):
        self.sigma = lst[0]
        self.eta0 = lst[1]
        self.kappa = lst[2]
        self.rho = lst[3]
        self.theta = lst[4]
        self.S0 = 282
        self.r = 0.015 
        self.q = 0.0177
        self.T = 1

    def Heston_cf(self, u): 
        """Heston model characteristic function """
        i = complex(0, 1)
        lam = np.sqrt(self.sigma**2*(u**2+i*u)+(self.kappa-i*self.rho*self.sigma*u)**2)
        w = np.exp(i*u*np.log(self.S0)+i*u*(self.r-self.q)*self.T+self.kappa*self.theta*self.T*\
        (self.kappa-i*self.rho*self.sigma*u)/self.sigma**2)/(np.cosh(lam*self.T/2)+(self.kappa-\
        i*self.rho*self.sigma*u)/lam*np.sinh(lam*self.T/2))**(2*self.kappa*self.theta/self.sigma**2)
        Psi = w*np.exp(-(u**2+i*u)*self.eta0/(lam/np.tanh(lam*self.T/2)+self.kappa-i*self.rho*self.sigma*u))
        return Psi

    def dirac(self, n):
        """define a dirac delta function"""
        y = np.zeros(len(n), dtype = complex)
        y[n==0] = 1
        return y

    def Heston_fft(self, alpha, n, upper_bound, K):
        N = 2 ** n
        delta_v = upper_bound / N
        delta_k = 2 * np.pi / N / delta_v

        J = m = np.arange(1, N+1, dtype = complex)

        Beta = np.log(self.S0) - delta_k * N / 2
        km = Beta + (m-1) * delta_k
        vj = (J-1) * delta_v #[nodes]
        i = complex(0,1)

        Psi_v = np.zeros(len(J), dtype = complex)
        for ii in range(N):
            u = vj[ii] - (alpha + 1) * i
            Psi_v[ii] = self.Heston_cf(u) / ((alpha + vj[ii] * i) * (alpha + 1 + vj[ii] * i))

        # compute FFT
        xj = (delta_v/2) * Psi_v * np.exp(-i * Beta * vj) * (2 - self.dirac(J-1))
        yj = np.fft.fft(xj)        
        
        # calculate option price
        CT_kj = np.exp(-alpha * np.array(km)) / np.pi * np.array(yj).real
        k_List = list(Beta + (np.cumsum(np.ones((N, 1))) - 1) * delta_k)
        Kt = np.exp(np.array(k_List)) # Real strike price K
        return np.exp(-self.r * self.T) * interpolate.splev(K, interpolate.splrep(Kt, np.real(CT_kj))).real
    
    def simulation(self, M, N, type, K1, K2):
        dt = 1 / M
        cov = dt * np.matrix([[1, self.rho], [self.rho, 1]])
        st = np.zeros((N, M))
        st[:, 0] = self.S0 
        vt = np.zeros((N, M))
        vt[:, 0] = self.eta0

        for i in range(1, M):
            w = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=N)
            st[:, i] = st[:, i - 1] + (self.r - self.q) * st[:, i - 1] * dt + np.array(np.sqrt(np.maximum(vt[:, i - 1], 0))) * (np.array(st[:, i - 1])) * w[:, 0]
            vt[:, i] = vt[:, i - 1] + self.kappa * (self.theta - vt[:, i - 1]) * dt + self.sigma * np.sqrt(np.maximum(vt[:, i - 1], 0)) * w[:, 1]

        if type == 'Euro':
            payoff = np.maximum(st[:, M - 1] - K1, 0)
            price = np.exp(-self.r) * np.mean(payoff)
            return price

        if type == 'up and out':
            payoff = np.maximum(st[:, M - 1] - K1, 0)
            payoff[st.max(axis=1) > K2] = 0
            price = np.exp(-self.r) * np.mean(payoff)
            return price

        if type == 'control variate':
            z = np.maximum(st[:, M - 1] - K1, 0)
            payoff = np.maximum(st[:, M - 1] - K1, 0)
            payoff[st.max(axis=1) > K2] = 0
            c = - np.cov(payoff, z)[0][1] / np.var(z)
            price = np.exp(-self.r) * np.mean(payoff + c * (z - np.mean(z)))
            return price

lst = params_star.x#[1.31, 0.046, 3.91, -0.79, 0.042]
K = 285
alpha = 1   
n = 15
upper_bound = 1000
N = 15000
M = 100
print('FFT Price:',FFT(lst).Heston_fft(alpha, n, upper_bound, K))

ht = 0.01

print('Inequality Satiefied?',ht > (1000/2000/1.31/1000) ** 2)

def path(S0, T, M, N, K, rho, sigma, kappa, theta, eta0):
    #one path
    #N paths, M time meshes
    ht = 1/M
    multi = np.random.multivariate_normal(mean = [0, 0], cov = ht * np.matrix([[1, rho], [rho, 1]]) , size = M)
    z1, z2 = multi[::, 0], multi[::, 1] #w1 and w2

    S, V = [S0], [eta0]
    for i in range(1, M):
        S += S[i-1] + (0.015 - 0.0177) * S[i-1] * ht + np.sqrt(max(V[i-1], 0)) * S[i-1] * z1[i],
        V += V[i-1] + kappa * (theta - V[i-1]) * ht + sigma * np.sqrt(max(V[i-1], 0)) * z2[i],
    return S
paths = [path(282, 1, M, N, K, lst[3], lst[0], lst[2], lst[4], lst[1]) for _ in range(N)]

def simu_euro(paths, N, K): 
    return sum( max(path[-1] - K, 0) * np.exp(-0.015 * 1) for path in paths) / N
print('Euro Price based on Simulation:', simu_euro(paths, N, K))

f = FFT(lst)

target = f.simulation(252, 50000, 'up and out', 285, 315)
N_list = [2000, 4000, 6000, 10000, 15000]
error = [target - f.simulation(252, i, 'up and out', 285, 315) for i in N_list]
np.array([N_list, error]).T

print('Up and Out Option Price based on Simulation:',target)

error2 = [target - f.simulation(252, i, 'control variate', 285, 315) for i in N_list]
result_df = pd.DataFrame(np.array([N_list, error2]).T)
print(result_df)








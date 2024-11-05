# coding=utf-8


import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

### ex. 1: Trapezio e Simpson
###         T = C*I
###         I = int(1/(np.sqrt(np.cos(teta) - np.cos(teta_o))))*d_teta
###             de teta = 0 até teta = teta_o
g = 9.8
def f(x, x_o):
    f = 1/(np.sqrt(np.cos(x) - np.cos(x_o)))
    return f

## Metodo do trapezio:

# Periodo em funcao do numero de pontos:
l = 1.
C = 4*np.sqrt(l/(2*g))
teta_o_deg = 40
teta_o = teta_o_deg*pi/180
T, num = [], []
for n in range(100, 10000, 100):
    I = 0
    a = 0
    b = teta_o
    x = np.linspace(a, b, n)
    for i in range(n - 1):
        if (np.cos(x[i]) - np.cos(b) > 0 and np.cos(x[i + 1]) - np.cos(b) > 0):
            I += (x[i + 1] - x[i])*(f(x[i], b) + f(x[i + 1], b))/2
    T += [C*I]
    num += [n]

plt.plot(num, T)
plt.xlabel(r"$n$")
plt.ylabel(r"$T\ (s)$")
plt.title(r"$Ex.1,\ m\'etodo\ do\ trap\'ezio:\ l\ =\ " + str(int(l)) + r"\ m,\ \theta_0\ =\ " +                  \
          str(int(teta_o_deg)) + r"\ \degree,\ g\ =\ 9.8\ m/s^2$")
plt.tight_layout()

# Periodo em funcao de l:
n = 10000
T, l_s = [], []
for l in np.arange(0, 5, 0.1):
    I = 0
    a = 0
    teta_o_deg = 40
    C = 4*np.sqrt(l/(2*g))
    b = teta_o_deg*pi/180
    x = np.linspace(a, b, n)
    for i in range(n - 1):
        if (np.cos(x[i]) - np.cos(b) > 0 and np.cos(x[i + 1]) - np.cos(b) > 0):
            I += (x[i + 1] - x[i])*(f(x[i], b) + f(x[i + 1], b))/2
    T += [C*I]
    l_s += [l]

plt.plot(l_s, T)
plt.xlabel(r"$l\ (m)$")
plt.ylabel(r"$T\ (s)$")
plt.title(r"$Ex.1,\ m\'etodo\ do\ trap\'ezio:\ \theta_0\ =\ 40 \degree,\ n\ = 10000,\ g\ =\ 9.8\ m/s^2$")
plt.tight_layout()

# Comparacao com a aproximacao de angulos pequenos:
l = 1
n = 10000
C = 4*np.sqrt(l/(2*g))
teta_dif_g, dif, P, T, tetas_o = [], [], [], [], []
for teta_o_deg in np.arange(0.1, 90, 0.5):
    I = 0
    a = 0
    b = teta_o_deg*pi/180
    x = np.linspace(a, b, n)
    for i in range(n - 1):
        if (np.cos(x[i]) - np.cos(b) > 0 and np.cos(x[i + 1]) - np.cos(b) > 0):
            I += (x[i + 1] - x[i])*(f(x[i], b) + f(x[i + 1], b))/2
    T += [C*I]
    tetas_o += [teta_o_deg]
    P += [2*pi*np.sqrt(l/g)]
    dif += [np.abs(100*(C*I - 2*pi*np.sqrt(l/g))/(2*pi*np.sqrt(l/g)))]
    if (np.abs(100*(C*I - 2*pi*np.sqrt(l/g))/(2*pi*np.sqrt(l/g))) > 1):
        teta_dif_g += [teta_o_deg]

fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel(r"$\theta_0\ (\degree)$")
ax1.set_ylabel(r"$T\ (s)$")

ax1.plot(tetas_o, T, color="C0", label=r"$integral$")
ax1.plot(tetas_o, P, color="C1", label=r"$aprox.\ de\ pequenos\ \theta_0$")
plt.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel(r"$\theta_0\ (\degree)$")
ax2.set_ylabel(r"$|Diferen c\c{} a|\ proporcional\ (\%)$")
ax2.plot([min(tetas_o), max(tetas_o)], [1, 1], color="C3", label=r"$limite" +  \
         "\ de\ 1\ \%$")
ax2.plot([min(teta_dif_g), min(teta_dif_g)], [min(dif), max(dif)/2.],          \
         color="C4", linestyle="--", label=r"$\theta_{limite}\ =\ " +          \
         str(round(min(teta_dif_g), 1)) + "\degree$")
plt.legend()

ax2.plot(tetas_o, dif, color="C2")

plt.suptitle(r"$Ex.1,\ m\'etodo\ do\ trap\'ezio:\ l\ =\ 1\ m,\ n\ = 10000,\ g\ =\ 9.8\ m/s^2$")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## Metdo do Simpson:

# Periodo em funcao do numero de pontos:
l = 1.
C = 4*np.sqrt(l/(2*g))
teta_o_deg = 40
teta_o = teta_o_deg*pi/180
T, num = [], []
for n in range(100, 10000, 100):
    I = 0
    a = 0
    b = teta_o
    m = 2*n
    x = np.linspace(a, b, m)
    h = x[1] - x[0]
    for i in range(n - 2):
        if (np.cos(x[2*i]) - np.cos(b) > 0 and                                 \
            np.cos(x[2*i + 1]) - np.cos(b) > 0 and                             \
            np.cos(x[2*i + 2]) - np.cos(b) > 0):
            I += (h/3)*(f(x[2*i], b) + 4*f(x[2*i + 1], b) + f(x[2*i + 2], b))
    T += [C*I]
    num += [n]

plt.plot(num, T)
plt.xlabel(r"$n$")
plt.ylabel(r"$T\ (s)$")
plt.title(r"$Ex.1,\ m\'etodo\ de\ Simpson:\ l\ =\ " + str(int(l)) + r"\ m,\ \theta_0\ =\ " +                  \
          str(int(teta_o_deg)) + r"\ \degree,\ g\ =\ 9.8\ m/s^2$")
plt.tight_layout()

# Periodo em funcao de l:
n = 10000
T, l_s = [], []
for l in np.arange(0, 5, 0.1):
    I = 0
    a = 0
    teta_o_deg = 40
    C = 4*np.sqrt(l/(2*g))
    b = teta_o_deg*pi/180
    m = 2*n
    x = np.linspace(a, b, m)
    h = x[1] - x[0]
    for i in range(n - 2):
        if (np.cos(x[2*i]) - np.cos(b) > 0 and \
            np.cos(x[2*i + 1]) - np.cos(b) > 0 and \
            np.cos(x[2*i + 2]) - np.cos(b) > 0):
            I += (h/3)*(f(x[2*i], b) + 4*f(x[2*i + 1], b) + f(x[2*i + 2], b))
    T += [C*I]
    l_s += [l]

plt.plot(l_s, T)
plt.xlabel(r"$l\ (m)$")
plt.ylabel(r"$T\ (s)$")
plt.title(r"$Ex.1,\ m\'etodo\ de\ Simpson:\ \theta_0\ =\ 40 \degree,\ n\ = 10000,\ g\ =\ 9.8\ m/s^2$")
plt.tight_layout()

# Comparacao com a aproximacao de angulos pequenos:
l = 1
n = 10000
C = 4*np.sqrt(l/(2*g))
teta_dif_p, teta_dif_g, dif, P, T, tetas_o = [], [], [], [], [], []
for teta_o_deg in np.arange(0.1, 90, 0.5):
    I = 0
    a = 0
    b = teta_o_deg*pi/180
    m = 2*n
    x = np.linspace(a, b, m)
    h = x[1] - x[0]
    for i in range(n - 2):
        if (np.cos(x[2*i]) - np.cos(b) > 0 and \
            np.cos(x[2*i + 1]) - np.cos(b) > 0 and \
            np.cos(x[2*i + 2]) - np.cos(b) > 0):
            I += (h/3)*(f(x[2*i], b) + 4*f(x[2*i + 1], b) + f(x[2*i + 2], b))
    T += [C*I]
    tetas_o += [teta_o_deg]
    P += [2*pi*np.sqrt(l/g)]
    dif += [np.abs(100*(C*I - 2*pi*np.sqrt(l/g))/(2*pi*np.sqrt(l/g)))]
    if (np.abs(100*(C*I - 2*pi*np.sqrt(l/g))/(2*pi*np.sqrt(l/g))) > 1 and \
        teta_o_deg > 10):
        teta_dif_g += [teta_o_deg]
    if (np.abs(100*(C*I - 2*pi*np.sqrt(l/g))/(2*pi*np.sqrt(l/g))) > 1 and \
        teta_o_deg < 10):
        teta_dif_p += [teta_o_deg]

fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel(r"$\theta_0\ (\degree)$")
ax1.set_ylabel(r"$T\ (s)$")

ax1.plot(tetas_o, T, color="C0", label=r"$integral$")
ax1.plot(tetas_o, P, color="C1", label=r"$aprox.\ de\ pequenos\ \theta_0$")
plt.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel(r"$\theta_0\ (\degree)$")
ax2.set_ylabel(r"$|Diferen c\c{} a|\ proporcional\ (\%)$")
ax2.plot([min(tetas_o), max(tetas_o)], [1, 1], color="C3", label=r"$limite" +  \
         "\ de\ 1\ \%$")
ax2.plot([max(teta_dif_p), max(teta_dif_p)], [min(dif), max(dif)/2.],          \
         color="C4", linestyle="--", label=r"$\theta_{max}\ =\ " +          \
         str(round(max(teta_dif_p), 1)) + "\degree$")
ax2.plot([min(teta_dif_g), min(teta_dif_g)], [min(dif), max(dif)/2.],          \
         color="C5", linestyle="--", label=r"$\theta_{min}\ =\ " +          \
         str(round(min(teta_dif_g), 1)) + "\degree$")
plt.legend()

ax2.plot(tetas_o, dif, color="C2")

plt.suptitle(r"$Ex.1,\ m\'etodo\ de\ Simpson:\ l\ =\ 1\ m,\ n\ = 10000,\ g\ =\ 9.8\ m/s^2$")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

### ex. 2: Romberg
###        I(x) = 0.5*I_o*((C(x) + 0.5)**2 + (S(x) + 0.5)**2)
###        C(x) = int(f1(t))*dt, de 0 a x
###        S(x) = int(f2(t))*dt, de 0 a x

## Metodo de Romberg:
def romberg(f, xi, xe, ni=20, mi=2):
    
    def R00():
        return 0.5*(xe - xi)*(f(xi) + f(xe))
    
    def Rn0(n):
        hn = (xe - xi)/(2.0**n)
        g = lambda k: f(xi + hn*(2.0*k - 1.0))
        summ = hn*sum([g(k) for k in xrange(1, 2**(n - 1))])
        return 0.5*R(n - 1, 0) + summ
    
    def Rnm(n, m):
        frac = 1.0/((4.0**m) - 1.0)
        rec1 = R(n, m - 1)
        rec2 = R(n - 1, m - 1)
        rec = (4.0**m)*rec1 - rec2
        return frac*rec
    
    def R(n, m):
        if n == 0 and m == 0:
            return R00()
        elif n > 0 and m == 0:
            return Rn0(n)
        else:
            return Rnm(n, m)
    
    return R(ni, mi)

def f1(t):
    return np.cos((pi*t**2)/2)

def f2(t):
    return np.sin((pi*t**2)/2)

I_o = 1 # W/m^2
x = np.linspace(0, 10, 1000)
I = []
for i in range(len(x)):
    C_x = romberg(f1, 0, x[i], ni=10, mi=10)
    S_x = romberg(f2, 0, x[i], ni=10, mi=10)
    I += [0.5*I_o*((C_x + 0.5)**2 + (S_x + 0.5)**2)]

plt.plot(x, I)
plt.xlabel(r"$x\ (m)$")
plt.ylabel(r"$I\ (W)$")
plt.tight_layout()


### ex. 3: Quadratura gaussiana
#   cv = 9*n_mol*k_bol*(1/u**3)*int((x**4*np.exp(x))/((np.exp(x) - 1)**2))*dx,
#                               de 0 a u

def f(x):
    return (x**4*np.exp(x))/((np.exp(x) - 1)**2)

# Metodo da quadratura gaussiana:
from scipy.integrate import fixed_quad

T = np.linspace(0.1, 3, 10000)

N2 = 2
cv2 = []
for i in range(len(T)):
    I = fixed_quad(f, a=0, b=1./T[i], n=N2)[0]
    cv2 += [(T[i]**3)*I]

N5 = 5
cv5 = []
for i in range(len(T)):
    I = fixed_quad(f, a=0, b=1./T[i], n=N5)[0]
    cv5 += [(T[i]**3)*I]

N10 = 10
cv10 = []
for i in range(len(T)):
    I = fixed_quad(f, a=0, b=1./T[i], n=N10)[0]
    cv10 += [(T[i]**3)*I]

T = np.copy(T)
cv2 = np.copy(cv2)
cv5 = np.copy(cv5)
cv10 = np.copy(cv10)

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 3, 1)
ax1.set_xlabel(r"$T/T_D$", size=14)
ax1.set_ylabel(r"$\frac{c_V}{9 \cdot n \cdot k}$", size=14)

ax1.plot(T, cv2, label=r"$N\ =\ 2$")
ax1.plot(T, cv5, label=r"$N\ =\ 5$")
ax1.plot(T, cv10, label=r"$N\ =\ 10$")
plt.legend()

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_xlabel(r"$T/T_D$", size=14)
ax2.set_ylabel(r"$Diferenc\c{}a$", size=14)
ax2.set_title(r"$Diferenc\c{}as$")

ax2.plot(T, cv10 - cv2, label=r"${c_V}_{n=10}\ -\ {c_V}_{n=2}$")
ax2.plot(T, cv10 - cv5, label=r"${c_V}_{n=10}\ -\ {c_V}_{n=5}$")
plt.legend()

ax3 = fig.add_subplot(1, 3, 3)
ax3.set_xlabel(r"$T/T_D$", size=14)
ax3.set_ylabel(r"$Diferenc\c{}a$", size=14)
ax3.set_title(r"$Diferenc\c{}as\ (zoom)$")

ax3.plot(T, cv10 - cv2, label=r"${c_V}_{n=10}\ -\ {c_V}_{n=2}$")
ax3.plot(T, cv10 - cv5, label=r"${c_V}_{n=10}\ -\ {c_V}_{n=5}$")
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


p = (pi/6)**(1/3.)
cv_einstein = ((p**2)/3.)*(1./T**2)*(np.exp(p/T))/((np.exp(p/T) - 1)**2)


plt.plot(T, cv10, label=r"$Debye\ (quadratura\ gaussiana: n\ =\ 10)$")
plt.plot(T, cv_einstein, label=r"$Einstein$")
plt.xlabel(r"$T/T_D$", size=14)
plt.ylabel(r"$\frac{c_V}{9 \cdot n \cdot k}$", size=14)
plt.legend()















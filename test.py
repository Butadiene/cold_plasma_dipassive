import numpy as np
import matplotlib.pyplot as plt
# 並列化用ライブラリ、引数操作用ライブラリをインポート
import multiprocessing
from functools import partial
from scipy.optimize import fsolve


#質量
#プラズマ周波数(電子のサイクロトロン周波数で規格化)
#サイクロトロン周波数(電子のサイクロトロン周波数で規格化)
# Constants
c = 3e8
me = 9.1093837015e-31
mh = 1.67262192e-27
a = 0.6 # proton and oxygen ratio
electron_cycltrons_seconds = 1

# Initialize plasma parameters
def init_plasma_params():
    oe, pe = 1.0*electron_cycltrons_seconds , 1.6*electron_cycltrons_seconds
    protons = {
        "mass": mh,
        "charge_sign": 1.0,
        "Cyclotron_freq": oe * me / mh,
        "Plasma_freq": pe * np.sqrt(a * me / mh),
        "name":"proton"
    }

    electrons = {
        "mass": me,
        "charge_sign": -1.0,
        "Cyclotron_freq": oe,
        "Plasma_freq": pe,
        "name": "electron"
    }
    return {"electron": electrons, "proton": protons}

elements = init_plasma_params()

if("electron" in elements):
  electron_plasma_freq = elements["electron"]["Plasma_freq"]
  electron_plasma_cycltrons = elements["electron"]["Cyclotron_freq"]
if("proton" in elements):
  proton_plasma_freq = elements["proton"]["Plasma_freq"]
  proton_plasma_cycltrons = elements["proton"]["Cyclotron_freq"]


# 分散関係の計算を関数化
def calc_disp(w,wave_num_tensor):
  K_perp = 1.0+0.0j
  K_x = 0.0j
  K_para = 1.0+0.0j

  for element in elements.values():
    plasma_frequencies = element["Plasma_freq"]
    plasma_cycltrons = element["Cyclotron_freq"]
    charge_sign = element["charge_sign"]
    K_perp -= plasma_frequencies**2.0/(w[0]**2.0-plasma_cycltrons**2.0)
    K_x += plasma_frequencies**2.0/(w[0]**2.0-plasma_cycltrons**2.0)*plasma_cycltrons*charge_sign/w[0]
    K_para -= plasma_frequencies**2.0/w[0]**2.0

  K_tensor = -np.array([[K_para,0.0j,0.0j],[0.0j,K_perp,-1.0j*K_x],[0.0j,1.0j*K_x,K_perp]])


  matrix = w[0]**2.0/c**2.0*K_tensor + wave_num_tensor

  det = np.linalg.det(matrix)

  return det 

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
ax1.set_ylim(1e-6,10)

ax1.set_xscale('log')
ax1.set_yscale('log')


array_num = 1000
k_array_for_index = np.logspace(-11.0,-7.0,array_num)
w_res = np.zeros(array_num)
w_res2 = np.zeros(array_num)
theta_list = np.arange(0,0.5*np.pi,0.1*np.pi)

Gamma_val = np.array([[[ 1.15313250e-11, 2.52422744e-01, 4.02169259e-32],
  [-2.52422765e-01, 1.02195665e-12,-1.66092652e-11],
  [ 1.38051800e-16, 1.66092652e-11, 1.12056538e-26]],

 [[-1.75308851e-11,-1.43615352e-01,-1.63412949e-16],
  [ 1.43615299e-01,-1.21972592e-11, 7.41904850e-12],
  [ 2.92722081e-16,-7.41904850e-12,-5.00535161e-27]],

 [[-8.14847673e-14,-1.35516567e-11,-1.44341128e-01],
  [-1.17565207e-08, 9.29697013e-13,-2.55996857e-02],
  [ 1.44341128e-01, 2.55996857e-02, 1.11024216e-12]]])

def upsilon(a,b,c):
  return Gamma_val[a][b][c] - Gamma_val[a][c][b]


for theta_index, theta in np.ndenumerate(theta_list):

  for k_index, k in np.ndenumerate(k_array_for_index):
    karray = np.array([k*np.cos(theta),k*np.sin(theta),0])

    wave_num_tensor = np.full((3, 3), 0.0+0.0j)
    for kappa in range(3):
      for beta in range(3):
        wave_num_tensor[kappa][kappa] += karray[beta]**2.0
        for tau in range(3):
          wave_num_tensor[kappa][kappa] += -1.0j*karray[beta]*upsilon(tau,beta,tau)

    for kappa in range(3):
      for neu in range(3):
        wave_num_tensor[kappa][neu] += -karray[kappa]*karray[neu]
        for tau in range(3):
          wave_num_tensor[kappa][neu] += +1.0j*karray[kappa]*upsilon(tau,neu,tau)
        for beta in range(3):
          wave_num_tensor[kappa][neu] +=1.0j*karray[beta]*(upsilon(beta,kappa,neu)+upsilon(neu,beta,kappa)) 
          for tau in range(3):
            wave_num_tensor[kappa][neu] += upsilon(beta,kappa,neu)*upsilon(tau,beta,tau)+0.5*upsilon(tau,beta,neu)*upsilon(tau,beta,kappa)    
    
    res = fsolve(calc_disp,1e-1,args = wave_num_tensor)
    res2 = fsolve(calc_disp,4e-4,args = wave_num_tensor,maxfev=10000)
    w_res[k_index] = res.real
    w_res2[k_index] = res2.real
  ax1.scatter(k_array_for_index,w_res)
  ax1.scatter(k_array_for_index,w_res2)
import matplotlib.pyplot as plt
import numpy as np
import control as ctrl

s = ctrl.TransferFunction.s


for alpha in [-0.01, 0, 0.01]:
    G = 1 / ((s+1)*(s+alpha))

    # Respuesta al escalón
    T, y = ctrl.step_response(G, T=np.linspace(0, 100, 1000))
    plt.figure(1)
    plt.subplot(2,1,1); plt.grid() 
    plt.plot(T, y, label=f'α = {alpha}')
    plt.legend()

    # Respuesta al escalón del sistema cerrado
    Gcl = G/(1+G)
    T, y = ctrl.step_response(Gcl,T=np.linspace(0, 10, 1000))
    plt.subplot(2,1,2); plt.grid()
    plt.plot(T, y, label=f'α = {alpha} (cerrado)')
    plt.legend()

    plt.figure(2)
    mag, phase, omega = ctrl.bode(G, dB=True, Hz=True, plot=True)

    plt.figure(3)
    ctrl.bode_plot(Gcl, dB=True, Hz=True, plot=True)


plt.figure(1)
plt.xlabel('Tiempo')
#plt.ylabel('Respuesta')
plt.subplot(2,1,1);plt.title('Respuesta al escalón para diferentes valores de α')

plt.show()
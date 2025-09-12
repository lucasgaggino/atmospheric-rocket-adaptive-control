import matplotlib.pyplot as plt
import numpy as np
import control as ctrl

s = ctrl.TransferFunction.s

for T in [0, 0.015, 0.030]:
    G = 400*(1-s*T) / ((s+1)*(s+20)*(1+s*T))

    # Respuesta al escalón
    t, y = ctrl.step_response(G, T=np.linspace(0, 5, 1000))
    plt.figure(1)
    plt.subplot(2,1,1); plt.grid() 
    plt.plot(t, y, label=f'T = {T}')    
    plt.legend()

    # Respuesta al escalón del sistema cerrado
    Gcl = G/(1+G)
    t, y = ctrl.step_response(Gcl,T=np.linspace(0, 1, 1000))
    plt.subplot(2,1,2); plt.grid()
    plt.plot(t, y, label=f'T = {T} (cerrado)')
    plt.legend()

    plt.figure(2)
    mag, phase, omega = ctrl.bode(G, dB=True, Hz=True, plot=True)

    plt.figure(3)
    ctrl.bode_plot(Gcl, dB=True, Hz=True, plot=True)


plt.figure(1)
plt.xlabel('Tiempo')
plt.subplot(2,1,1);plt.title('Respuesta al escalón para diferentes valores de T')

plt.figure(4)
ctrl.pzmap(G, plot=True)

plt.show()


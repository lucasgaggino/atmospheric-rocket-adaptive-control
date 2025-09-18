import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
import control as ctl


# Parametros físicos (modificables)
M = 1.0  # masa del carro [kg]
m = 0.1  # masa del péndulo [kg]
l = 0.5  # semilongitud de la barra [m] (centro de masa)
g = 9.81  # gravedad [m/s^2]
F_const = 1.0  # fuerza aplicada al carro [N] (constante)


# Dinamica no lineal (sin rozamientos)
# Estado: y = [x, xdot, theta, thetadot]
# Modelo clasico con barra uniforme (I = (1/3) m l^2)
def cartpole_ode(t, y):
    x, xdot, th, thdot = y
    s, c = math.sin(th), math.cos(th)

    # Auxiliar
    temp = (F_const + m * l * thdot * thdot * s) / (M + m)

    # Denominador con inercia de barra uniforme
    denom = l * (4.0 / 3.0 - (m * c * c) / (M + m))

    # Aceleraciones
    thddot = (g * s - c * temp) / denom
    xddot = temp - (m * l * thddot * c) / (M + m)

    return [xdot, xddot, thdot, thddot]


# Condicion inicial
y0 = [0.0, 0.0, 0.0, 0.0]  # x, xdot, theta, thetadot

# Integracion
T = 5.0
t_eval = np.linspace(0.0, T, 1001)
sol = solve_ivp(cartpole_ode, [0.0, T], y0, t_eval=t_eval)

# Graficar evolucion angular
theta = sol.y[2]
print(f"θ(T={T:.2f} s) = {theta[-1]:.6f} rad  ({theta[-1]*180/math.pi:.3f}°)")

plt.plot(sol.t, theta)
plt.xlabel("t [s]")
plt.ylabel("θ [rad]")
plt.title("Péndulo invertido (entrada F = 1 N, no lineal)")
plt.grid(True, alpha=0.3)
plt.show()


###### Linealizacion ######

A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))  # >0 (inestable)
B_theta = -3.0 / (l * (4.0 * M + m))

# La planta linealizada (entrada F, salida θ) en Laplace:
# Θ(s)/F(s) = B_theta / (s^2 - A_theta)
num_c = [B_theta]
den_c = [1.0, 0.0, -A_theta]
G_s = ctl.TransferFunction(num_c, den_c)  # G(s)

# -------------------------------------------
# DISCRETIZACIoN (ZOH) -> H(z)
# -------------------------------------------
Ts = 0.02  # tiempo de muestreo [s]
G_z_zoh = ctl.c2d(G_s, Ts, method="zoh")

H_z = G_z_zoh

# como obtener los coeficientes de la transferencia discreta
# num_d = np.squeeze(G_z_zoh.num)  # vector 1D
# den_d = np.squeeze(G_z_zoh.den)
# H_z = ctl.TransferFunction(num_d, den_d, Ts)


print("G(s) continuo (linealizado)   :", G_s)
print("H(z) discreto (ZOH, Ts=%.3f s):" % Ts, H_z)

# -------------------------------------------
# RESPUESTA AL ESCALON: continuo vs discreto
# -------------------------------------------
t_final = 1.0

# Continuo
t_cont = np.linspace(0, t_final, 1000)
t_cont, y_cont = ctl.step_response(G_s, T=t_cont)

# Discreto (mismas convenciones que tu ejemplo)
t_disc = np.arange(0, t_final + 1e-12, Ts)
t_disc, y_disc = ctl.step_response(H_z, T=t_disc)

# -------------------------------------------
# GRAFICOS
# -------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(t_cont, y_cont, label="G(s) continuo")
plt.plot(t_disc, y_disc, "o", ms=3, label="H(z) ZOH (discreto)")
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("t [s]")
plt.ylabel(r"$\theta$ [rad]")
plt.title("Respuesta al escalon: continuo vs discreto (entrada F)")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------
# DIAGRAMAS DE POLOS Y CEROS
# -------------------------------------------

plt.figure(figsize=(12, 5))

# Sistema Continuo
plt.subplot(1, 2, 1)
plt.title("Sistema Continuo - G(s)", fontsize=11, fontweight="bold")
poles_c, zeros_c = ctl.pole_zero_map(G_s)
print("Polos de G(s):", poles_c)
print("Ceros de G(s):", zeros_c)
plt.scatter(
    np.real(poles_c), np.imag(poles_c), marker="x", s=100, color="red", label="Polos"
)
if len(zeros_c) > 0:
    plt.scatter(
        np.real(zeros_c),
        np.imag(zeros_c),
        marker="o",
        s=100,
        color="blue",
        label="Ceros",
        facecolors="none",
    )
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginaria")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")

# Sistema Discreto
plt.subplot(1, 2, 2)
plt.title("Sistema Discreto - H(z) ZOH", fontsize=11, fontweight="bold")
poles_d, zeros_d = ctl.pole_zero_map(H_z)
print("Polos de H(z):", poles_d)
print("Ceros de H(z):", zeros_d)
plt.scatter(
    np.real(poles_d), np.imag(poles_d), marker="x", s=100, color="red", label="Polos"
)
if len(zeros_d) > 0:
    plt.scatter(
        np.real(zeros_d),
        np.imag(zeros_d),
        marker="o",
        s=100,
        color="blue",
        label="Ceros",
        facecolors="none",
    )

# circulo unitario
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5, label="Círculo Unitario")

plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginaria")
plt.grid(True, alpha=0.3)
plt.axis("equal")

plt.suptitle("Diagramas de Polos y Ceros", fontsize=12, fontweight="bold", y=1.08)

plt.show()

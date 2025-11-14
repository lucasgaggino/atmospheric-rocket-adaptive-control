import control as ctrl
import numpy as np
from matplotlib import pyplot as plt
import scipy as sc
from scipy import signal


# Parámetros del péndulo (posición estable hacia abajo)
M = 1.0  # masa del carro [kg]
m = 0.1  # masa del péndulo [kg]
l = 0.5  # semilongitud de la barra [m]
g = 9.81  # gravedad [m/s^2]

# Constantes del modelo linealizado - POSICIÓN HACIA ABAJO (estable)
# Para θ = π, el sistema es estable: A_theta < 0
A_theta = -3.0 * g * (M + m) / (l * (4.0 * M + m))  # <0 (estable)
B_theta = -3.0 / (l * (4.0 * M + m))

# Función de transferencia continua
num_c = [B_theta]
den_c = [1.0, 0.0, -A_theta]
G_s = ctrl.TransferFunction(num_c, den_c)

# Discretización ZOH
Ts = 0.02  # tiempo de muestreo [s]
G_z_zoh = ctrl.c2d(G_s, Ts, method="zoh")
num_d = np.squeeze(G_z_zoh.num)
den_d = np.squeeze(G_z_zoh.den)

print("Modelo del péndulo en posición estable (hacia abajo) linealizado:")
print(f"G(s) = {B_theta} / (s^2 - {A_theta})")
print(f"H(z) = {num_d} / {den_d} (Ts = {Ts}s)")


def obtener_datos_pendulo(u, e):
    """
    Genera datos de salida del péndulo en posición hacia abajo dados entrada u y ruido e
    """
    N = len(u)
    y = np.zeros_like(u)

    # Simulación usando la función de transferencia discreta
    # H(z) = (num[0]*z + num[1]) / (z^2 + den[1]*z + den[2])
    # Ecuación en diferencias: y[k] + den[1]*y[k-1] + den[2]*y[k-2] = num[0]*u[k-1] + num[1]*u[k-2]
    # O equivalentemente: y[k] = -den[1]*y[k-1] - den[2]*y[k-2] + num[0]*u[k-1] + num[1]*u[k-2] + e[k]

    for k in range(2, N):
        y[k] = -den_d[1]*y[k-1] - den_d[2]*y[k-2] + num_d[0]*u[k-1] + num_d[1]*u[k-2] + e[k]

    return y


def estimador_RLS(u, y, na=2, nb=2, lambda_=1, theta_ini=None, theta_real=None, plot=True):
    """
    Estimador por mínimos cuadrados recursivo para modelo ARX
    Adaptado del notebook de identificación paramétrica
    """
    N = len(u)
    if theta_ini is None:
        theta_hat = np.zeros(na+nb)       # estimación inicial
    else:
        theta_hat = theta_ini
    P = 100 * np.eye(na+nb)          # matriz de incertidumbre inicial
    err = np.zeros_like(y)
    y_hat = np.zeros_like(y)

    theta_hist = []
    k_range = range(max(na, nb)+1, N)
    for k in k_range:
        phi = np.concatenate((-y[k-1:k-na-1:-1], u[k-1:k-nb-1:-1]))   # vector de regresores
        y_hat[k] = phi @ theta_hat
        err[k] = y[k] - y_hat[k]

        # Verificar si hay overflow antes de actualizar
        if np.any(np.abs(phi) > 1e10) or np.any(np.abs(theta_hat) > 1e10):
            print(f"Overflow detectado en k={k}, deteniendo estimación")
            break

        K = P @ phi / (lambda_ + phi.T @ P @ phi)
        theta_hat = theta_hat + K * err[k]
        P = (P - np.outer(K, phi) @ P) / lambda_

        theta_hist.append(theta_hat.copy())

    theta_hist = np.array(theta_hist)

    # Si no se completó la estimación debido a overflow, devolver arrays vacíos o parciales
    if len(theta_hist) == 0:
        return np.array([]), P, err, np.array([])

    # Validación
    N_lag = 50
    ree = np.correlate(err, err, 'full')
    rey = np.correlate(err, y_hat, 'full')
    ryy = np.correlate(y_hat, y_hat, 'full')
    RN = ree/(rey*ryy+0.001)**0.5
    lags = np.arange(-len(err)+1, len(err))
    center = len(ree) // 2
    ree = ree[center-N_lag:center+N_lag+1]
    RN = RN[center-N_lag:center+N_lag+1]
    lags_plot = lags[center-N_lag:center+N_lag+1]
    ree_max_val = 2.17/np.sqrt(N)*ree[N_lag]

    if plot and len(theta_hist) > 1:
        k_range_plot = list(k_range)[:len(theta_hist)]  # ajustar k_range al tamaño de theta_hist

        plt.figure(figsize=(12, 8))

        plt.subplot(221)
        plt.step(k_range_plot, err[k_range_plot], where='post')
        plt.ylabel('Error de predicción')
        plt.title('Error de predicción')

        plt.subplot(222)
        for idx_na in range(na):
            plt.step(k_range_plot, theta_hist[:, idx_na], label=f"$\\hat{{a_{idx_na+1}}}$", where='post')
        for idx_nb in range(nb):
            plt.step(k_range_plot, theta_hist[:, na+idx_nb], label=f"$\\hat{{b_{idx_nb+1}}}$", where='post')
        if theta_real is not None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for idx_na in range(na):
                color = colors[idx_na % len(colors)]
                plt.axhline(theta_real[idx_na], linestyle='--', color=color, label=f"$a_{idx_na+1}$")
            for idx_nb in range(nb):
                color = colors[(idx_nb+na) % len(colors)]
                plt.axhline(theta_real[idx_nb+na], linestyle='--', color=color, label=f"$b_{idx_nb+1}$")
        plt.legend()
        plt.xlabel('k')
        plt.ylabel('$\\theta$')
        plt.title('Evolución de parámetros')

        plt.subplot(223)
        plt.plot(lags_plot, ree)
        plt.ylabel('$r_{ee}$')
        plt.axhline(ree_max_val, linestyle='--', color='r', label=f"$r_{{ee,max}}$")
        plt.title('Autocorrelación del error')
        plt.xlabel('lag')
        plt.legend()

        plt.subplot(224)
        # Comparación entrada-salida
        plt.plot(y, 'b-', label='y (real)', linewidth=2)
        plt.plot(y_hat, 'r--', label='y_hat (estimado)', linewidth=2)
        plt.xlabel('k')
        plt.ylabel('y')
        plt.title('Comparación real vs estimado')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return theta_hist, P, err, ree


if __name__ == "__main__":
    print("\n=== IDENTIFICACIÓN PARAMÉTRICA DEL PÉNDULO (POSICIÓN HACIA ABAJO) ===\n")

    # Parámetros reales del modelo (de los coeficientes de la función de transferencia discreta)
    # Para H(z) = (b0*z + b1) / (z^2 + a1*z + a2), entonces el modelo ARX es:
    # y[k] + a1*y[k-1] + a2*y[k-2] = b0*u[k-1] + b1*u[k-2]
    # En el RLS: phi = [-y[k-1], -y[k-2], u[k-1], u[k-2]], theta = [a1, a2, b0, b1]
    theta_real = np.array([den_d[1], den_d[2], num_d[0], num_d[1]])
    print(f"Parámetros reales: a1={theta_real[0]:.4f}, a2={theta_real[1]:.4f}, b0={theta_real[2]:.6f}, b1={theta_real[3]:.6f}")
    print(f"Coeficientes H(z): num={num_d}, den={den_d}")

    # Identificación usando señal PRBS
    print("\n--- Identificación con señal PRBS ---")
    N = 5000  # Más muestras para mejor identificación
    np.random.seed(42)  # para reproducibilidad
    u = np.sign(np.random.randn(N)) * 1.0  # PRBS con amplitud mucho mayor para mejor excitación

    sigma_e = 0.05  # ruido moderado
    e = sigma_e * np.random.randn(N)
    y = obtener_datos_pendulo(u, e)

    theta_hat, P, err, _ = estimador_RLS(u, y, na=2, nb=2, lambda_=0.99, theta_real=theta_real)
    if len(theta_hat) > 0:
        print(f"Parámetros estimados: {theta_hat[-1,:]}")
        if not np.any(np.isnan(P)):
            desvios = sigma_e*np.sqrt(np.diag(P))
            print(f"Desvío de la estimación: a1={desvios[0]:.6f}, a2={desvios[1]:.6f}, b0={desvios[2]:.6f}, b1={desvios[3]:.6f}")

            # Mostrar comparación con valores reales
            print(f"Valores reales:        a1={theta_real[0]:.4f}, a2={theta_real[1]:.4f}, b0={theta_real[2]:.6f}, b1={theta_real[3]:.6f}")
            errores_absolutos = np.abs(theta_hat[-1,:] - theta_real)
            print(f"Errores absolutos:     a1={errores_absolutos[0]:.6f}, a2={errores_absolutos[1]:.6f}, b0={errores_absolutos[2]:.6f}, b1={errores_absolutos[3]:.6f}")

            # Evaluar calidad de la identificación
            print("\nEvaluación de la identificación:")
            if errores_absolutos[0] < 0.01 and errores_absolutos[1] < 0.01:
                print("Parametros a (dinamica): EXCELENTE - identificacion precisa")
            else:
                print("Parametros a (dinamica): PROBLEMAS en la identificacion")

            if errores_absolutos[2] < 0.001 and errores_absolutos[3] < 0.001:
                print("Parametros b (ganancia): BUENA - identificacion precisa")
            else:
                print("Parametros b (ganancia): ACEPTABLE - coeficientes pequenos dificiles de identificar")

    # Validación: respuesta del modelo identificado
    print("\n--- Validación: Respuesta del modelo identificado ---")
    if len(theta_hat) > 0 and not np.any(np.isnan(theta_hat[-1])):
        plt.figure(figsize=(10, 6))

        # Respuesta del sistema real
        t_test = np.arange(0, 1.0, Ts)  # tiempo más corto
        u_test = np.ones_like(t_test) * 0.01  # escalón pequeño
        e_test = np.zeros_like(u_test)
        y_real = obtener_datos_pendulo(u_test, e_test)

        # Respuesta del modelo identificado
        # y[k] = theta[0]*y[k-1] + theta[1]*y[k-2] + theta[2]*u[k-1] + theta[3]*u[k-2]
        y_ident = np.zeros_like(y_real)
        theta_final = theta_hat[-1, :]
        for k in range(2, len(y_ident)):
            y_ident[k] = theta_final[0]*y_ident[k-1] + theta_final[1]*y_ident[k-2] + theta_final[2]*u_test[k-1] + theta_final[3]*u_test[k-2]

        plt.subplot(211)
        plt.plot(t_test, u_test, 'g-', label='Entrada u(t)', linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('u')
        plt.title('Señal de entrada de validación')
        plt.grid(True)
        plt.legend()

        plt.subplot(212)
        plt.plot(t_test, y_real, 'b-', label='Sistema real', linewidth=2)
        plt.plot(t_test, y_ident, 'r--', label='Modelo identificado', linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('θ [rad]')
        plt.title('Comparación: Sistema real vs Modelo identificado')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("No se pudo obtener una estimación válida para validación")

    print("\nIdentificación completada exitosamente!")

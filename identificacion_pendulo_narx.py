import control as ctrl
import numpy as np
from matplotlib import pyplot as plt
import scipy as sc
from scipy import signal
import os


# Parámetros del péndulo (posición vertical - inestable)
M = 1.0  # masa del carro [kg]
m = 0.1  # masa del péndulo [kg]
l = 0.5  # semilongitud de la barra [m]
g = 9.81  # gravedad [m/s^2]

# Constantes del modelo linealizado - POSICIÓN VERTICAL (inestable)
# Para θ = π, el sistema es inestable: A_theta > 0
A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))  # >0 (inestable)
B_theta = 3.0 / (l * (4.0 * M + m))

# Función de transferencia continua
num_c = [B_theta]
den_c = [1.0, 0.0, -A_theta]
G_s = ctrl.TransferFunction(num_c, den_c)

# Discretización ZOH
Ts = 0.02  # tiempo de muestreo [s]
G_z_zoh = ctrl.c2d(G_s, Ts, method="zoh")
num_d = np.squeeze(G_z_zoh.num)
den_d = np.squeeze(G_z_zoh.den)

# Crear carpeta para guardar imágenes
output_dir = "imagenes_identificacion_narx"
os.makedirs(output_dir, exist_ok=True)

print("Modelo del péndulo en posición vertical (inestable) - NARX:")
print(f"G(s) = {B_theta} / (s^2 - {A_theta})")
print(f"H(z) = {num_d} / {den_d} (Ts = {Ts}s)")


def obtener_datos_pendulo(u, e):
    """
    Genera datos de salida del péndulo en posición vertical dados entrada u y ruido e
    """
    N = len(u)
    y = np.zeros_like(u)

    # Simulación usando la función de transferencia discreta
    # H(z) = (num[0]*z + num[1]) / (z^2 + den[1]*z + den[2])
    # Ecuación en diferencias: y[k] + den[1]*y[k-1] + den[2]*y[k-2] = num[0]*u[k-1] + num[1]*u[k-2] + e[k]

    for k in range(2, N):
        y[k] = -den_d[1]*y[k-1] - den_d[2]*y[k-2] + num_d[0]*u[k-1] + num_d[1]*u[k-2] + e[k]

    return y


def estimador_NARX(u, y, na=1, nb=1, nl=1, lambda_=1, theta_ini=None, plot=True):
    """
    Estimador por mínimos cuadrados recursivo para modelo NARX
    y[k] = θ₀*y[k-1] + θ₁*sin(y[k-1]) + θ₂*u[k-1] + θ₃*u[k-1]² + e[k]

    Parámetros: [θ₀, θ₁, θ₂, θ₃]
    """
    N = len(u)
    n_theta = na + nl + nb + nl  # [y[k-1], sin(y[k-1]), u[k-1], u[k-1]²]

    if theta_ini is None:
        theta_hat = np.zeros(n_theta)
    else:
        theta_hat = theta_ini

    P = 100 * np.eye(n_theta)
    err = np.zeros_like(y)
    y_hat = np.zeros_like(y)

    theta_hist = []
    k_range = range(max(na, nb, nl)+1, N)

    for k in k_range:
        # Vector de regresores para NARX: [y[k-1], sin(y[k-1]), u[k-1], u[k-1]²]
        phi = np.array([
            y[k-1],                    # θ₀: término lineal en y
            np.sin(y[k-1]),           # θ₁: término no lineal sin(y)
            u[k-1],                   # θ₂: término lineal en u
            u[k-1]**2                 # θ₃: término no lineal u²
        ])

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

    # Si no se completó la estimación debido a overflow, devolver arrays vacíos
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
        k_range_plot = list(k_range)[:len(theta_hist)]

        plt.figure(figsize=(12, 8))

        plt.subplot(221)
        plt.step(k_range_plot, err[k_range_plot], where='post')
        plt.ylabel('Error de predicción')
        plt.title('Error de predicción')

        plt.subplot(222)
        param_names = ['$\\theta_0$', '$\\theta_1$', '$\\theta_2$', '$\\theta_3$']
        colors = ['blue', 'red', 'green', 'orange']
        for i in range(n_theta):
            plt.step(k_range_plot, theta_hist[:, i], color=colors[i], label=param_names[i], where='post')
        plt.legend()
        plt.xlabel('k')
        plt.ylabel('$\\theta$')
        plt.title('Evolución de parámetros NARX')

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
        plt.title('Comparación real vs NARX')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'narx_identification.png'), dpi=300, bbox_inches='tight')
        plt.show()

    return theta_hist, P, err, ree


if __name__ == "__main__":
    print("\n=== IDENTIFICACIÓN NARX DEL PÉNDULO INESTABLE ===\n")

    # Identificación usando señal PRBS con amplitud muy pequeña para estabilidad
    print("\n--- Identificación NARX con señal PRBS (sistema inestable) ---")
    N = 3000  # Más muestras para NARX
    np.random.seed(42)
    u = np.sign(np.random.randn(N)) * 0.02  # PRBS con amplitud MUY pequeña
    # Sistema inestable requiere excitación muy pequeña

    sigma_e = 0.001  # ruido muy reducido
    e = sigma_e * np.random.randn(N)
    y = obtener_datos_pendulo(u, e)

    # Parámetros del modelo NARX
    theta_narx, P, err, _ = estimador_NARX(u, y, na=1, nb=1, nl=1, lambda_=0.85)

    if len(theta_narx) > 0:
        print(f"Parámetros NARX identificados: {theta_narx[-1,:]}")
        if not np.any(np.isnan(P)):
            desvios = sigma_e*np.sqrt(np.diag(P))
            print(f"Desvio de la estimacion NARX: theta0={desvios[0]:.6f}, theta1={desvios[1]:.6f}, theta2={desvios[2]:.6f}, theta3={desvios[3]:.6f}")

        # Validación con la misma señal PRBS
        print("\n--- Validación NARX con PRBS ---")
        N_test = min(800, N)
        u_test = u[:N_test] * 0.5  # misma señal pero amplitud moderada para testing
        e_test = np.zeros(N_test)
        y_real_test = obtener_datos_pendulo(u_test, e_test)

        # Simulación NARX
        y_narx_test = np.zeros_like(y_real_test)
        theta_final = theta_narx[-1, :]
        overflow_detected = False

        for k in range(1, len(y_narx_test)):
            phi_test = np.array([
                y_narx_test[k-1],          # y[k-1]
                np.sin(y_narx_test[k-1]), # sin(y[k-1])
                u_test[k-1],               # u[k-1]
                u_test[k-1]**2             # u[k-1]²
            ])

            y_new = phi_test @ theta_final

            if np.abs(y_new) > 1e10 or not np.isfinite(y_new):
                print(f"Overflow en validación NARX en k={k}")
                overflow_detected = True
                break

            y_narx_test[k] = y_new

        if not overflow_detected:
            mse_narx = np.mean((y_real_test - y_narx_test)**2)
            print(f"MSE validación NARX: {mse_narx:.2e}")
            print(f"RMSE validación NARX: {np.sqrt(mse_narx):.2e}")

            # Comparación con modelo lineal ARMAX
            plt.figure(figsize=(12, 6))

            plt.subplot(121)
            plt.plot(y_real_test[:300], 'b-', label='Sistema real', linewidth=2)
            plt.plot(y_narx_test[:300], 'r--', label='NARX identificado', linewidth=2)
            plt.xlabel('k')
            plt.ylabel('theta [rad]')
            plt.title('Validación NARX: Sistema Inestable')
            plt.legend()
            plt.grid(True)

            plt.subplot(122)
            plt.plot(u_test[:300], 'g-', label='Entrada PRBS', linewidth=2)
            plt.xlabel('k')
            plt.ylabel('u')
            plt.title('Señal de excitación')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'narx_validation.png'), dpi=300, bbox_inches='tight')
            plt.show()

        # Análisis de convergencia de parámetros NARX
        plt.figure(figsize=(12, 8))

        k_range_plot = np.arange(theta_narx.shape[0])

        param_names = ['$\\theta_0$ (lineal)', '$\\theta_1$ (sin)', '$\\theta_2$ (u)', '$\\theta_3$ (u²)']
        colors = ['blue', 'red', 'green', 'orange']

        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.plot(k_range_plot, theta_narx[:, i], color=colors[i], linewidth=2)
            plt.ylabel(param_names[i])
            plt.xlabel('Iteración k')
            plt.title(f'Convergencia {param_names[i]}')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'narx_parameters.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Comparación NARX vs Lineal
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.plot(y_real_test[:200], 'k-', label='Real', linewidth=3, alpha=0.7)
        plt.plot(y_narx_test[:200], 'r-', label='NARX', linewidth=2)
        plt.xlabel('k')
        plt.ylabel('θ [rad]')
        plt.title('NARX vs Real: Captura de No Linealidades')
        plt.legend()
        plt.grid(True)

        plt.subplot(122)
        error_narx = y_real_test - y_narx_test
        plt.plot(error_narx[:200], 'r-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.title('Error de Predicción NARX')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'narx_vs_linear.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print("\n--- Evaluación NARX ---")
        print("Parámetros identificados:")
        print(f"theta0 = {theta_final[0]:.6f}")
        print(f"theta1 = {theta_final[1]:.6f}")
        print(f"theta2 = {theta_final[2]:.6f}")
        print(f"theta3 = {theta_final[3]:.6f}")

        # Comparar con dinamica teorica
        print("\nComparacion con dinamica esperada:")
        print(f"theta0 deberia capturar dinamica lineal ~ {den_d[1]}")
        print(f"theta1 deberia capturar no linealidad sin(theta) (esperado != 0)")
        print(f"theta2 deberia capturar ganancia lineal ~ {num_d[0]}")
        print(f"theta3 deberia capturar no linealidad u^2 (posible != 0)")

        print("\n--- Conclusion NARX ---")
        if abs(theta_narx[-1, 1]) > 1e-4:  # Si theta1 != 0 (con tolerancia)
            print("ÉXITO: NARX captura no linealidades del sistema inestable")
            print("El termino sin(theta) es significativo, confirmando modelado no lineal")
        else:
            print("Limitado: NARX no captura no linealidades significativas")

    else:
        print("No se pudo identificar modelo NARX válido")

    print("\nIdentificación NARX completada!")

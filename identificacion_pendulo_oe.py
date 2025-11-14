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
output_dir = "imagenes_identificacion_oe"
os.makedirs(output_dir, exist_ok=True)

print("Modelo del pendulo en posicion vertical (inestable) - Output Error:")
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


def estimador_OutputError(u, y, na=2, nb=2, lambda_=1, theta_ini=None, plot=True):
    """
    Estimador por mínimos cuadrados recursivo para modelo Output Error
    y[k] = B(q)/F(q) u[k] + e[k]

    Parámetros: [f1, f2, b0, b1] donde F(q) = 1 + f1*q^-1 + f2*q^-2, B(q) = b0 + b1*q^-1
    """
    N = len(u)
    n_theta = na + nb  # [f1, f2, b0, b1]

    if theta_ini is None:
        theta_hat = np.zeros(n_theta)
    else:
        theta_hat = theta_ini

    P = 100 * np.eye(n_theta)
    err = np.zeros_like(y)
    y_hat = np.zeros_like(y)

    theta_hist = []
    k_range = range(max(na, nb)+1, N)

    for k in k_range:
        # Simulación one-step-ahead usando parámetros actuales
        # y_hat[k] = B(q)/F(q) u[k] (sin ruido)
        # Para k actual, usamos u[k] y u[k-1]
        # y_hat[k] = b0*u[k] + b1*u[k-1] - f1*y_hat[k-1] - f2*y_hat[k-2]

        # Para predicción one-step-ahead, necesitamos simular desde el principio con parámetros actuales
        y_sim = np.zeros(k+1)
        for j in range(max(na, nb), k+1):
            # Simulación del modelo con parámetros actuales
            y_sim[j] = (theta_hat[na]*u[j] + theta_hat[na+1]*u[j-1] -
                       theta_hat[0]*y_sim[j-1] - theta_hat[1]*y_sim[j-2])

        y_hat[k] = y_sim[k]
        err[k] = y[k] - y_hat[k]

        # Verificar si hay overflow antes de actualizar
        if np.any(np.abs(y_sim) > 1e10) or np.any(np.abs(theta_hat) > 1e10):
            print(f"Overflow detectado en k={k}, deteniendo estimacion")
            break

        # Gradiente para OE: ∂ε/∂θ = -∂ŷ/∂θ
        # Para OE, el gradiente es más complejo porque requiere simulación
        # Usamos aproximación simplificada
        phi = np.zeros(n_theta)

        # Derivadas aproximadas (simplificadas)
        # ∂ŷ/∂f1 = -y_hat[k-1], ∂ŷ/∂f2 = -y_hat[k-2]
        # ∂ŷ/∂b0 = u[k], ∂ŷ/∂b1 = u[k-1]
        phi[0] = -y_hat[k-1] if k > 0 else 0
        phi[1] = -y_hat[k-2] if k > 1 else 0
        phi[2] = u[k]
        phi[3] = u[k-1] if k > 0 else 0

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
        plt.ylabel('Error de prediccion')
        plt.title('Error de prediccion OE')

        plt.subplot(222)
        param_names = ['$f_1$', '$f_2$', '$b_0$', '$b_1$']
        colors = ['blue', 'red', 'green', 'orange']
        for i in range(n_theta):
            plt.step(k_range_plot, theta_hist[:, i], color=colors[i], label=param_names[i], where='post')
        plt.legend()
        plt.xlabel('k')
        plt.ylabel('theta')
        plt.title('Evolucion de parametros OE')

        plt.subplot(223)
        plt.plot(lags_plot, ree)
        plt.ylabel('r_ee')
        plt.axhline(ree_max_val, linestyle='--', color='r', label=f"r_ee,max")
        plt.title('Autocorrelacion del error')
        plt.xlabel('lag')
        plt.legend()

        plt.subplot(224)
        # Comparación entrada-salida
        plt.plot(y, 'b-', label='y (real)', linewidth=2)
        plt.plot(y_hat, 'r--', label='y_hat (estimado)', linewidth=2)
        plt.xlabel('k')
        plt.ylabel('y')
        plt.title('Comparacion real vs OE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'oe_identification.png'), dpi=300, bbox_inches='tight')
        plt.show()

    return theta_hist, P, err, ree


if __name__ == "__main__":
    print("\n=== IDENTIFICACION OUTPUT ERROR DEL PENDULO INESTABLE ===\n")

    # Identificación usando señal PRBS con amplitud muy pequeña para estabilidad
    print("\n--- Identificacion OE con senal PRBS (sistema inestable) ---")
    N = 3000  # Más muestras para OE
    np.random.seed(42)
    u = np.sign(np.random.randn(N)) * 0.02  # PRBS con amplitud MUY pequeña
    # Sistema inestable requiere excitación muy pequeña

    sigma_e = 0.001  # ruido muy reducido
    e = sigma_e * np.random.randn(N)
    y = obtener_datos_pendulo(u, e)

    # Parámetros del modelo OE
    theta_oe, P, err, _ = estimador_OutputError(u, y, na=2, nb=2, lambda_=0.85)

    if len(theta_oe) > 0:
        print(f"Parametros OE identificados: {theta_oe[-1,:]}")
        if not np.any(np.isnan(P)):
            desvios = sigma_e*np.sqrt(np.diag(P))
            print(f"Desvio de la estimacion OE: f1={desvios[0]:.6f}, f2={desvios[1]:.6f}, b0={desvios[2]:.6f}, b1={desvios[3]:.6f}")

        # Validación con la misma señal PRBS
        print("\n--- Validacion OE con PRBS ---")
        N_test = min(800, N)
        u_test = u[:N_test] * 0.5  # misma senal pero amplitud moderada para testing
        e_test = np.zeros(N_test)
        y_real_test = obtener_datos_pendulo(u_test, e_test)

        # Simulación OE completa con parámetros finales
        theta_final = theta_oe[-1, :]
        y_oe_test = np.zeros_like(y_real_test)

        for k in range(max(2, 2), len(y_oe_test)):
            y_oe_test[k] = (theta_final[2]*u_test[k] + theta_final[3]*u_test[k-1] -
                           theta_final[0]*y_oe_test[k-1] - theta_final[1]*y_oe_test[k-2])

        # Verificar si la simulación fue exitosa
        overflow_detected = np.any(np.abs(y_oe_test) > 1e10) or np.any(np.isnan(y_oe_test))

        if not overflow_detected:
            mse_oe = np.mean((y_real_test - y_oe_test)**2)
            print(f"MSE validacion OE: {mse_oe:.2e}")
            print(f"RMSE validacion OE: {np.sqrt(mse_oe):.2e}")

            # Comparación con modelo lineal
            plt.figure(figsize=(12, 6))

            plt.subplot(121)
            plt.plot(y_real_test[:300], 'b-', label='Sistema real', linewidth=2)
            plt.plot(y_oe_test[:300], 'r--', label='OE identificado', linewidth=2)
            plt.xlabel('k')
            plt.ylabel('theta [rad]')
            plt.title('Validacion OE: Sistema Inestable')
            plt.legend()
            plt.grid(True)

            plt.subplot(122)
            plt.plot(u_test[:300], 'g-', label='Entrada PRBS', linewidth=2)
            plt.xlabel('k')
            plt.ylabel('u')
            plt.title('Senal de excitacion')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'oe_validation.png'), dpi=300, bbox_inches='tight')
            plt.show()

        # Análisis de convergencia de parámetros OE
        plt.figure(figsize=(12, 8))

        k_range_plot = np.arange(theta_oe.shape[0])

        param_names = ['$f_1$ (dinamico)', '$f_2$ (dinamico)', '$b_0$ (entrada)', '$b_1$ (entrada)']
        colors = ['blue', 'red', 'green', 'orange']

        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.plot(k_range_plot, theta_oe[:, i], color=colors[i], linewidth=2)
            plt.ylabel(param_names[i])
            plt.xlabel('Iteracion k')
            plt.title(f'Convergencia {param_names[i]}')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'oe_parameters.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Comparación OE vs Real
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.plot(y_real_test[:200], 'k-', label='Real', linewidth=3, alpha=0.7)
        plt.plot(y_oe_test[:200], 'r-', label='OE', linewidth=2)
        plt.xlabel('k')
        plt.ylabel('theta [rad]')
        plt.title('OE vs Real: Modelo Lineal OE')
        plt.legend()
        plt.grid(True)

        plt.subplot(122)
        error_oe = y_real_test - y_oe_test
        plt.plot(error_oe[:200], 'r-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.title('Error de Prediccion OE')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'oe_vs_linear.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print("\n--- Evaluacion OE ---")
        print("Parametros identificados:")
        print(f"f1 = {theta_final[0]:.6f}")
        print(f"f2 = {theta_final[1]:.6f}")
        print(f"b0 = {theta_final[2]:.6f}")
        print(f"b1 = {theta_final[3]:.6f}")

        # Comparar con dinámica teórica
        print("\nComparacion con dinamica esperada:")
        print(f"f1 deberia capturar dinamica lineal ~ {den_d[1]}")
        print(f"f2 deberia capturar dinamica lineal ~ {den_d[2]}")
        print(f"b0 deberia capturar ganancia ~ {num_d[0]}")
        print(f"b1 deberia ser pequeno (ZOH) ~ {num_d[1]}")

        print("\n--- Conclusion OE ---")
        if not overflow_detected:
            print("EXITO: OE converge sin overflow en sistema inestable")
            print("El modelo lineal OE es adecuado para ruido aditivo de medicion")
        else:
            print("Limitado: OE presenta overflow en validacion")

    else:
        print("No se pudo identificar modelo OE valido")

    print("\nIdentificacion OE completada!")

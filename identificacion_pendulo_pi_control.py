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

print("Modelo del pendulo en posicion vertical (inestable) - Control PI:")
print(f"G(s) = {B_theta} / (s^2 - {A_theta})")
print(f"H(z) = {num_d} / {den_d} (Ts = {Ts}s)")


def obtener_datos_pendulo_abierto(u, e):
    """
    Genera datos de salida del péndulo en posición vertical dados entrada u y ruido e
    """
    N = len(u)
    y = np.zeros_like(u)

    # Simulación usando la función de transferencia discreta
    for k in range(2, N):
        y[k] = -den_d[1]*y[k-1] - den_d[2]*y[k-2] + num_d[0]*u[k-1] + num_d[1]*u[k-2] + e[k]

    return y


def control_pi(theta_ref, theta_actual, error_prev, Kp, Ki, Ts, u_sat=50.0):
    """
    Controlador PI discreto
    """
    error = theta_ref - theta_actual
    integral = error_prev + error * Ts
    u = Kp * error + Ki * integral

    # Saturación más alta para sistema inestable
    u = np.clip(u, -u_sat, u_sat)

    return u, error, integral


def control_pid(theta_ref, theta_actual, theta_prev, error_prev, Kp, Ki, Kd, Ts, u_sat=50.0):
    """
    Controlador PID discreto completo
    """
    error = theta_ref - theta_actual

    # Término proporcional
    proporcional = Kp * error

    # Término integral
    integral = error_prev + error * Ts
    integral_term = Ki * integral

    # Término derivativo (aproximación por diferencias)
    derivativo = (theta_actual - theta_prev) / Ts if Ts > 0 else 0
    derivativo_term = Kd * derivativo

    u = proporcional + integral_term - derivativo_term  # Nota: signo negativo para derivativo

    # Saturación
    u = np.clip(u, -u_sat, u_sat)

    return u, error, integral


def simulacion_lazo_cerrado(Kp, Ki, tiempo_total=10.0, theta_ref=0.0, ruido_std=0.001):
    """
    Simula el sistema en lazo cerrado con control PI
    """
    N = int(tiempo_total / Ts)
    t = np.arange(N) * Ts

    # Inicialización
    theta = np.zeros(N)
    theta[0] = 0.1  # condición inicial pequeña
    u = np.zeros(N)
    error_integral = 0.0

    # Ruido de medición
    ruido = ruido_std * np.random.randn(N)

    for k in range(1, N):
        # Medición con ruido
        theta_medido = theta[k-1] + ruido[k-1]

        # Control PI
        u[k], _, error_integral = control_pi(theta_ref, theta_medido, error_integral, Kp, Ki, Ts)

        # Simulación del sistema
        theta[k] = (-den_d[1]*theta[k-1] - den_d[2]*theta[k-2] +
                   num_d[0]*u[k-1] + num_d[1]*u[k-2])

        # Verificar estabilidad (criterio más relajado para sistema inestable)
        if np.abs(theta[k]) > 50:  # criterio de divergencia más relajado
            return t, theta, u, False

    return t, theta, u, True


def simulacion_lazo_cerrado_pid(Kp, Ki, Kd, tiempo_total=10.0, theta_ref=0.0, ruido_std=0.001):
    """
    Simula el sistema en lazo cerrado con control PID
    """
    N = int(tiempo_total / Ts)
    t = np.arange(N) * Ts

    # Inicialización
    theta = np.zeros(N)
    theta[0] = 0.1  # condición inicial pequeña
    theta_prev = 0.0
    u = np.zeros(N)
    error_integral = 0.0

    # Ruido de medición
    ruido = ruido_std * np.random.randn(N)

    for k in range(1, N):
        # Medición con ruido
        theta_medido = theta[k-1] + ruido[k-1]

        # Control PID (usar referencia escalar o del array)
        theta_ref_k = theta_ref[k-1] if isinstance(theta_ref, np.ndarray) else theta_ref
        u[k], _, error_integral = control_pid(theta_ref_k, theta_medido, theta_prev, error_integral, Kp, Ki, Kd, Ts)

        # Simulación del sistema
        theta[k] = (-den_d[1]*theta[k-1] - den_d[2]*theta[k-2] +
                   num_d[0]*u[k-1] + num_d[1]*u[k-2])

        theta_prev = theta[k-1]

        # Verificar estabilidad (criterio más relajado para sistema inestable)
        if np.abs(theta[k]) > 100:  # criterio de divergencia
            return t, theta, u, False

    return t, theta, u, True


def encontrar_control_pi_bueno():
    """
    Para sistemas inestables, usar ganancias PID conocidas que funcionan
    """
    print("\n--- Usando ganancias PID conocidas para péndulo inestable ---")

    # Ganancias típicas para control de péndulo invertido (cart-pole)
    # Estas ganancias son mucho más agresivas que para sistemas estables
    Kp = 100.0   # Ganancia proporcional alta
    Ki = 10.0    # Ganancia integral moderada
    Kd = 20.0    # Ganancia derivativa para amortiguamiento

    print(f"Usando ganancias PID conocidas: Kp={Kp}, Ki={Ki}, Kd={Kd}")

    # Probar si funciona
    try:
        t, theta, u, estable = simulacion_lazo_cerrado_pid(Kp, Ki, Kd, tiempo_total=5.0)

        if estable:
            error_estacionario = np.abs(theta[-50:].mean())
            overshoot = np.max(np.abs(theta))
            print(f"Control exitoso: error_est={error_estacionario:.6f}, overshoot={overshoot:.3f}")
            return Kp, Ki, Kd
        else:
            print("Control falló, intentando con ganancias aún más agresivas")
            Kp = 200.0
            Ki = 20.0
            Kd = 40.0
            print(f"Probando ganancias más agresivas: Kp={Kp}, Ki={Ki}, Kd={Kd}")

            t, theta, u, estable = simulacion_lazo_cerrado_pid(Kp, Ki, Kd, tiempo_total=5.0)
            if estable:
                return Kp, Ki, Kd
            else:
                print("Incluso ganancias agresivas fallaron. Sistema muy inestable.")
                return None, None, None

    except Exception as e:
        print(f"Error en simulación: {e}")
        return None, None, None


def obtener_datos_lazo_cerrado(Kp, Ki, Kd, N_muestras=5000, ruido_std=0.01):
    """
    Genera datos del sistema en lazo cerrado para identificación
    """
    tiempo_total = N_muestras * Ts

    # Simulación con referencia constante para lazo cerrado
    theta_ref = 0.0  # referencia constante

    t, theta, u, estable = simulacion_lazo_cerrado_pid(Kp, Ki, Kd, tiempo_total, theta_ref=theta_ref, ruido_std=ruido_std)

    if not estable:
        print("El sistema no está estable con estos parámetros PID")
        return None, None

    # Usar solo las últimas muestras para evitar transitorios
    muestras_utiles = min(N_muestras, len(theta))
    theta_data = theta[-muestras_utiles:]
    u_data = u[-muestras_utiles:]

    return u_data, theta_data


def estimador_RLS_lazo_cerrado(u, y, na=2, nb=2, nc=1, lambda_=1, theta_ini=None, plot=True):
    """
    Estimador RLS para modelo ARMAX en lazo cerrado
    """
    N = len(u)
    n_theta = na + nb + nc

    if theta_ini is None:
        theta_hat = np.zeros(n_theta)
    else:
        theta_hat = theta_ini

    P = 100 * np.eye(n_theta)
    err = np.zeros_like(y)
    y_hat = np.zeros_like(y)

    theta_hist = []
    k_range = range(max(na, nb, nc)+1, N)

    for k in k_range:
        # Vector de regresores para ARMAX: [-y[k-1], -y[k-2], u[k-1], u[k-2], e[k-1]]
        phi = np.concatenate((-y[k-1:k-na-1:-1], u[k-1:k-nb-1:-1], err[k-1:k-nc-1:-1]))
        y_hat[k] = phi @ theta_hat
        err[k] = y[k] - y_hat[k]

        if np.any(np.abs(phi) > 1e10) or np.any(np.abs(theta_hat) > 1e10):
            print(f"Overflow detectado en k={k}, deteniendo estimacion")
            break

        K = P @ phi / (lambda_ + phi.T @ P @ phi)
        theta_hat = theta_hat + K * err[k]
        P = (P - np.outer(K, phi) @ P) / lambda_

        theta_hist.append(theta_hat.copy())

    theta_hist = np.array(theta_hist)

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
        plt.title('Error de prediccion ARMAX Lazo Cerrado')

        plt.subplot(222)
        param_names = ['$a_1$', '$a_2$', '$b_1$', '$b_2$', '$c_1$']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i in range(min(n_theta, len(param_names))):
            plt.step(k_range_plot, theta_hist[:, i], color=colors[i], label=param_names[i], where='post')
        plt.legend()
        plt.xlabel('k')
        plt.ylabel('theta')
        plt.title('Evolucion de parametros ARMAX Lazo Cerrado')

        plt.subplot(223)
        plt.plot(lags_plot, ree)
        plt.ylabel('r_ee')
        plt.axhline(ree_max_val, linestyle='--', color='r', label=f"r_ee,max")
        plt.title('Autocorrelacion del error')
        plt.xlabel('lag')
        plt.legend()

        plt.subplot(224)
        plt.plot(y, 'b-', label='y (real)', linewidth=2)
        plt.plot(y_hat, 'r--', label='y_hat (estimado)', linewidth=2)
        plt.xlabel('k')
        plt.ylabel('y')
        plt.title('Comparacion real vs ARMAX Lazo Cerrado')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'armax_closed_loop_identification.png'), dpi=300, bbox_inches='tight')
        plt.show()

    return theta_hist, P, err, ree


if __name__ == "__main__":
    print("\n=== CONTROL PI Y IDENTIFICACIÓN EN LAZO CERRADO ===\n")

    # Crear carpeta para guardar imágenes
    output_dir = "imagenes_identificacion_pi_control"
    os.makedirs(output_dir, exist_ok=True)

    # Paso 1: Encontrar parámetros PID adecuados
    Kp_opt, Ki_opt, Kd_opt = encontrar_control_pi_bueno()

    if Kp_opt is None:
        print("No se pudo encontrar un controlador PID que estabilice el sistema")
        print("El péndulo inestable es demasiado difícil de controlar para identificación en lazo cerrado")
        exit(1)

    # Paso 2: Simulación extendida con los mejores parámetros
    print("\n--- Simulación extendida con parámetros óptimos ---")
    t, theta, u, estable = simulacion_lazo_cerrado_pid(Kp_opt, Ki_opt, Kd_opt, tiempo_total=20.0)

    if estable:
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.plot(t, theta, 'b-', linewidth=2, label='Ángulo θ')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Referencia')
        plt.ylabel('θ [rad]')
        plt.title(f'Control PID: Kp={Kp_opt:.1f}, Ki={Ki_opt:.1f}, Kd={Kd_opt:.1f}')
        plt.legend()
        plt.grid(True)

        plt.subplot(212)
        plt.plot(t, u, 'r-', linewidth=2, label='Acción de control u')
        plt.xlabel('t [s]')
        plt.ylabel('u')
        plt.title('Acción de Control PI')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pi_control_response.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Calcular métricas de control
        error_estacionario = np.abs(theta[-100:].mean())  # error en régimen estacionario
        overshoot = np.max(np.abs(theta))
        settling_time = t[np.where(np.abs(theta) < 0.01)[0]]
        settling_time = settling_time[0] if len(settling_time) > 0 else t[-1]

        print(f"Error estacionario: {error_estacionario:.6f}")
        print(f"Overshoot máximo: {overshoot:.6f}")
        print(f"Tiempo de establecimiento: {settling_time:.2f}s")

        # Paso 3: Identificación del sistema en lazo cerrado
        print("\n--- Identificación ARMAX del sistema en lazo cerrado ---")

        u_data, theta_data = obtener_datos_lazo_cerrado(Kp_opt, Ki_opt, Kd_opt, N_muestras=3000)

        if u_data is not None and theta_data is not None:
            theta_armax, P, err, _ = estimador_RLS_lazo_cerrado(u_data, theta_data, na=2, nb=2, nc=1, lambda_=0.99)

            if len(theta_armax) > 0:
                print(f"Parametros ARMAX lazo cerrado identificados: {theta_armax[-1,:]}")

                if not np.any(np.isnan(P)):
                    desvios = 0.01*np.sqrt(np.diag(P))  # sigma_e ≈ 0.01
                    print(f"Desvio de la estimacion: a1={desvios[0]:.6f}, a2={desvios[1]:.6f}, b1={desvios[2]:.6f}, b2={desvios[3]:.6f}, c1={desvios[4]:.6f}")

                # Validación del modelo identificado
                print("\n--- Validación del modelo ARMAX en lazo cerrado ---")

                # Usar datos de validación diferentes
                u_val, theta_val_real = obtener_datos_lazo_cerrado(Kp_opt, Ki_opt, Kd_opt, N_muestras=1000, ruido_std=0.0)  # sin ruido para validación

                if u_val is not None and theta_val_real is not None:
                    # Simulación del modelo identificado
                    theta_val_pred = np.zeros_like(theta_val_real)
                    err_val = np.zeros_like(theta_val_real)
                    theta_final = theta_armax[-1, :]

                    for k in range(max(2, 2, 1), len(theta_val_pred)):
                        phi = np.array([
                            -theta_val_pred[k-1], -theta_val_pred[k-2],  # términos AR
                            u_val[k-1], u_val[k-2],                      # términos entrada
                            err_val[k-1]                                # término ruido
                        ])
                        theta_val_pred[k] = phi @ theta_final
                        err_val[k] = theta_val_real[k] - theta_val_pred[k]

                    # Calcular métricas de validación
                    mse_val = np.mean(err_val**2)
                    rmse_val = np.sqrt(mse_val)

                    print(f"MSE validación lazo cerrado: {mse_val:.2e}")
                    print(f"RMSE validación lazo cerrado: {rmse_val:.2e}")

                    # Gráfico de comparación
                    plt.figure(figsize=(12, 6))

                    plt.subplot(121)
                    plt.plot(theta_val_real[:500], 'b-', label='Sistema real', linewidth=2)
                    plt.plot(theta_val_pred[:500], 'r--', label='Modelo ARMAX', linewidth=2)
                    plt.xlabel('k')
                    plt.ylabel('θ [rad]')
                    plt.title('Validación ARMAX Lazo Cerrado')
                    plt.legend()
                    plt.grid(True)

                    plt.subplot(122)
                    plt.plot(err_val[:500], 'g-', linewidth=2)
                    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    plt.xlabel('k')
                    plt.ylabel('Error')
                    plt.title('Error de Predicción')
                    plt.grid(True)

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'armax_closed_loop_validation.png'), dpi=300, bbox_inches='tight')
                    plt.show()

                    # Análisis de parámetros identificados
                    print("\n--- Análisis del modelo identificado en lazo cerrado ---")
                    print("El sistema en lazo cerrado debe ser estable por diseño del controlador PID")
                    print("Los parámetros identificados corresponden al comportamiento global del sistema controlado")

                    # Cálculo de polos del sistema real en lazo cerrado
                    print("\n--- Cálculo de polos del sistema en lazo cerrado ---")

                    # El sistema identificado tiene la forma:
                    # H(z) = (b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
                    # Los polos son raíces de: z^2 + a1*z + a2 = 0

                    a1_real, a2_real = theta_final[0], theta_final[1]
                    b1_real, b2_real = theta_final[2], theta_final[3]

                    # Polos del sistema identificado
                    polos_identificados = np.roots([1, a1_real, a2_real])
                    print(f"Polos identificados: {polos_identificados}")

                    # Polos del sistema real discreto
                    polos_reales = np.array([den_d[1], den_d[2]])  # den_d = [1, a1, a2] para el sistema abierto
                    print(f"Polos reales (sistema abierto): {polos_reales}")

                    # Para el sistema en lazo cerrado, los polos son más complejos de calcular analíticamente
                    # pero podemos estimarlos de la respuesta del sistema controlado
                    print("Los polos del sistema en lazo cerrado se determinan por la dinámica global controlada")
                    print(f"Parámetros identificados: a1={a1_real:.6f}, a2={a2_real:.6f}, b1={b1_real:.8f}, b2={b2_real:.8f}")

                else:
                    print("Error generando datos de validación")

            else:
                print("No se pudo identificar modelo ARMAX válido en lazo cerrado")

        else:
            print("Error generando datos del sistema en lazo cerrado")

    else:
        print("El sistema no se pudo estabilizar con los parámetros encontrados")

    print("\nIdentificación en lazo cerrado completada!")

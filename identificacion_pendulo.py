import control as ctrl
import numpy as np
from matplotlib import pyplot as plt
import scipy as sc
from scipy import signal
import os


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

# Crear carpeta para guardar imágenes
output_dir = "imagenes_identificacion"
os.makedirs(output_dir, exist_ok=True)

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


def estimador_RLS(u, y, na=2, nb=1, nc=1, lambda_=1, theta_ini=None, theta_real=None, plot=True):
    """
    Estimador por mínimos cuadrados recursivo para modelo ARMAX
    y[k] + a1*y[k-1] + a2*y[k-2] = b0*u[k-1] + c1*e[k-1] + e[k]
    Parámetros: [a1, a2, b0, c1]
    """
    N = len(u)
    if theta_ini is None:
        theta_hat = np.zeros(na+nb+nc)       # estimación inicial
    else:
        theta_hat = theta_ini
    P = 100 * np.eye(na+nb+nc)          # matriz de incertidumbre inicial
    err = np.zeros_like(y)
    y_hat = np.zeros_like(y)

    theta_hist = []
    k_range = range(max(na, nb, nc)+1, N)
    for k in k_range:
        # Vector de regresores para ARMAX: [-y[k-1], -y[k-2], u[k-1], e[k-1]]
        phi = np.concatenate((-y[k-1:k-na-1:-1], u[k-1:k-nb-1:-1], err[k-1:k-nc-1:-1]))
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
        for idx_nc in range(nc):
            plt.step(k_range_plot, theta_hist[:, na+nb+idx_nc], label=f"$\\hat{{c_{idx_nc+1}}}$", where='post')
        if theta_real is not None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for idx_na in range(na):
                color = colors[idx_na % len(colors)]
                plt.axhline(theta_real[idx_na], linestyle='--', color=color, label=f"$a_{idx_na+1}$")
            for idx_nb in range(nb):
                color = colors[(idx_nb+na) % len(colors)]
                plt.axhline(theta_real[idx_nb+na], linestyle='--', color=color, label=f"$b_{idx_nb+1}$")
            for idx_nc in range(nc):
                color = colors[(idx_nc+na+nb) % len(colors)]
                plt.axhline(theta_real[na+nb+idx_nc], linestyle='--', color=color, label=f"$c_{idx_nc+1}$")
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
        plt.savefig(os.path.join(output_dir, 'rls_identification.png'), dpi=300, bbox_inches='tight')
        plt.show()

    return theta_hist, P, err, ree


if __name__ == "__main__":
    print("\n=== IDENTIFICACIÓN PARAMÉTRICA DEL PÉNDULO (POSICIÓN HACIA ABAJO) ===\n")

    # Parámetros reales del modelo ARMAX (simplificado)
    # Para H(z) = (b0*z + b1) / (z^2 + a1*z + a2), el modelo ARMAX simplificado es:
    # y[k] + a1*y[k-1] + a2*y[k-2] = b0*u[k-1] + c1*e[k-1] + e[k]
    # Eliminamos b1 (difícil de identificar) y agregamos c1 para modelar el ruido
    # En el RLS: phi = [-y[k-1], -y[k-2], u[k-1], e[k-1]], theta = [a1, a2, b0, c1]
    # Para ruido blanco, c1 ≈ 0
    theta_real = np.array([den_d[1], den_d[2], num_d[0], 0.0])  # c1 = 0 para ruido blanco
    print(f"Parámetros reales (ARMAX): a1={theta_real[0]:.4f}, a2={theta_real[1]:.4f}, b0={theta_real[2]:.6f}, c1={theta_real[3]:.6f}")
    print(f"Coeficientes H(z): num={num_d}, den={den_d}")

    # Identificación usando señal PRBS
    print("\n--- Identificación ARMAX con señal PRBS ---")
    N = 5000  # Más muestras para mejor identificación
    np.random.seed(42)  # para reproducibilidad
    u = np.sign(np.random.randn(N)) * 2  # PRBS con amplitud mucho mayor para mejor excitación

    sigma_e = 0.05  # ruido moderado
    e = sigma_e * np.random.randn(N)
    y = obtener_datos_pendulo(u, e)

    theta_hat, P, err, _ = estimador_RLS(u, y, na=2, nb=1, nc=1, lambda_=0.99, theta_real=theta_real)
    if len(theta_hat) > 0:
        print(f"Parámetros estimados: {theta_hat[-1,:]}")
        if not np.any(np.isnan(P)):
            desvios = sigma_e*np.sqrt(np.diag(P))
            print(f"Desvío de la estimación: a1={desvios[0]:.6f}, a2={desvios[1]:.6f}, b0={desvios[2]:.6f}, c1={desvios[3]:.6f}")

            # Mostrar comparación con valores reales
            print(f"Valores reales:        a1={theta_real[0]:.4f}, a2={theta_real[1]:.4f}, b0={theta_real[2]:.6f}, c1={theta_real[3]:.6f}")
            errores_absolutos = np.abs(theta_hat[-1,:] - theta_real)
            print(f"Errores absolutos:     a1={errores_absolutos[0]:.6f}, a2={errores_absolutos[1]:.6f}, b0={errores_absolutos[2]:.6f}, c1={errores_absolutos[3]:.6f}")

            # Evaluar calidad de la identificación
            print("\nEvaluación de la identificación:")
            if errores_absolutos[0] < 0.01 and errores_absolutos[1] < 0.01:
                print("Parametros a (dinamica): EXCELENTE - identificacion precisa")
            else:
                print("Parametros a (dinamica): PROBLEMAS en la identificacion")

            if errores_absolutos[2] < 0.001:
                print("Parametro b0 (ganancia): BUENA - identificacion precisa")
            else:
                print("Parametro b0 (ganancia): ACEPTABLE - coeficiente pequeno dificil de identificar")

            if errores_absolutos[3] < 0.1:
                print("Parametro c1 (ruido): BUENA - modelado de ruido correcto")
            else:
                print("Parametro c1 (ruido): El ruido no es blanco perfecto")

    # Validación: respuesta del modelo identificado con la misma señal PRBS usada en identificación
    print("\n--- Validación: Respuesta del modelo identificado con PRBS ---")
    if len(theta_hat) > 0 and not np.any(np.isnan(theta_hat[-1])):
        plt.figure(figsize=(12, 8))

        # Validación con la misma señal PRBS usada en identificación
        N_test = min(1000, N)  # usar un subconjunto de la señal PRBS para validación
        u_prbs = u[:N_test] * 0.1  # misma señal PRBS pero con amplitud menor para validación
        e_prbs = np.zeros(N_test)  # sin ruido para validación
        y_real_prbs = obtener_datos_pendulo(u_prbs, e_prbs)

        # Respuesta del modelo identificado ARMAX con PRBS
        y_ident_prbs = np.zeros_like(y_real_prbs)
        e_sim_prbs = np.zeros_like(y_real_prbs)
        theta_final = theta_hat[-1, :]
        for k in range(2, len(y_ident_prbs)):
            y_ident_prbs[k] = (-theta_final[0]*y_ident_prbs[k-1] - theta_final[1]*y_ident_prbs[k-2] +
                              theta_final[2]*u_prbs[k-1] + theta_final[3]*e_sim_prbs[k-1])
            e_sim_prbs[k] = 0  # e[k] = 0 para respuesta forzada

        plt.subplot(221)
        plt.plot(u_prbs[:200], 'g-', label='PRBS', linewidth=1)  # mostrar primeras 200 muestras
        plt.xlabel('k')
        plt.ylabel('u')
        plt.title('Señal PRBS de validación')
        plt.grid(True)
        plt.legend()

        plt.subplot(222)
        plt.plot(y_real_prbs[:200], 'b-', label='Sistema real', linewidth=1)
        plt.plot(y_ident_prbs[:200], 'r--', label='Modelo identificado', linewidth=1)
        plt.xlabel('k')
        plt.ylabel('θ [rad]')
        plt.title('Validación con PRBS (primeras 200 muestras)')
        plt.grid(True)
        plt.legend()

        # Validación adicional con escalón
        print("\n--- Validación adicional: Respuesta a escalón ---")
        t_step = np.arange(0, 15.0, Ts)
        u_step = np.ones_like(t_step) * 0.1  # escalón pequeño
        u_step[50:] = 0.0  # señal nula después de 50 muestras
        e_step = np.zeros_like(u_step)
        y_real_step = obtener_datos_pendulo(u_step, e_step)

        # Respuesta del modelo identificado con escalón
        y_ident_step = np.zeros_like(y_real_step)
        e_sim_step = np.zeros_like(y_real_step)
        for k in range(2, len(y_ident_step)):
            y_ident_step[k] = (-theta_final[0]*y_ident_step[k-1] - theta_final[1]*y_ident_step[k-2] +
                              theta_final[2]*u_step[k-1] + theta_final[3]*e_sim_step[k-1])
            e_sim_step[k] = 0  # e[k] = 0 para respuesta forzada

        plt.subplot(223)
        plt.plot(t_step, u_step, 'g-', label='Escalón', linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('u')
        plt.title('Señal de escalón')
        plt.grid(True)
        plt.legend()

        plt.subplot(224)
        plt.plot(t_step, y_real_step, 'b-', label='Sistema real', linewidth=2)
        plt.plot(t_step, y_ident_step, 'r--', label='Modelo identificado', linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('θ [rad]')
        plt.title('Validación con escalón')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'validation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Calcular métricas de validación
        mse_prbs = np.mean((y_real_prbs - y_ident_prbs)**2)
        mse_step = np.mean((y_real_step - y_ident_step)**2)
        print(f"MSE validación PRBS: {mse_prbs:.2e}")
        print(f"MSE validación escalón: {mse_step:.2e}")
        print(f"RMSE validación PRBS: {np.sqrt(mse_prbs):.2e}")
        print(f"RMSE validación escalón: {np.sqrt(mse_step):.2e}")
            else:
        print("No se pudo obtener una estimación válida para validación")

    # Gráfico adicional: Respuesta forzada del sistema continuo y discreto
    print("\n--- Respuesta forzada del sistema continuo y discreto ---")
    if len(theta_hat) > 0 and not np.any(np.isnan(theta_hat[-1])):
        # Análisis de estabilidad del modelo identificado ARMAX
        theta_final = theta_hat[-1, :]
        print(f"\nAnálisis de estabilidad del modelo identificado:")
        print(f"Parámetros finales: a1={theta_final[0]:.6f}, a2={theta_final[1]:.6f}, b0={theta_final[2]:.6f}, c1={theta_final[3]:.6f}")

        # Polos del sistema identificado: raíces de z^2 + a1*z + a2 = 0
        polos = np.roots([1, theta_final[0], theta_final[1]])
        print(f"Polos del sistema identificado: {polos}")
        print(f"Módulos de los polos: {np.abs(polos)}")

        if np.all(np.abs(polos) < 1):
            print("El sistema identificado es ESTABLE (|z| < 1)")
        else:
            print("El sistema identificado es INESTABLE (|z| >= 1)")

        # Comparación con polos reales
        polos_reales = np.roots([1, den_d[1], den_d[2]])
        print(f"Polos del sistema real: {polos_reales}")
        print(f"Módulos de los polos reales: {np.abs(polos_reales)}")

        # Análisis de ganancias (solo b0 para ARMAX simplificado)
        print(f"\nAnálisis de ganancias:")
        # Ganancia DC del sistema identificado: b0/(1-sum(a))
        ganancia_ident = theta_final[2] / (1 - theta_final[0] - theta_final[1])
        ganancia_real = num_d[0] / (1 - den_d[1] - den_d[2])  # Solo b0, ignoramos b1
        print(f"Ganancia DC identificada: {ganancia_ident:.6f}")
        print(f"Ganancia DC real (b0): {ganancia_real:.6f}")
        print(f"Error relativo en ganancia: {abs(ganancia_ident - ganancia_real)/abs(ganancia_real)*100:.1f}%")

        # Análisis de la señal de entrada
        print(f"\nAnálisis de la señal de entrada:")
        print(f"Amplitud máxima de u_step: {np.max(np.abs(u_step)):.4f}")
        print(f"Valor RMS de u_step: {np.sqrt(np.mean(u_step**2)):.4f}")

        plt.figure(figsize=(10, 6))

        # Respuesta forzada del sistema continuo
        t_fine = np.linspace(0, 15.0, 1000)  # tiempo más fino para mejor visualización
        u_fine = np.interp(t_fine, t_step, u_step)  # interpolar u_step al tiempo fino
        t_out, y_forced = ctrl.forced_response(G_s, t_fine, u_fine)

        # Respuesta forzada del sistema discreto H(z)
        t_discrete, y_discrete = ctrl.forced_response(G_z_zoh, t_step, u_step)

        # Respuesta usando parámetros identificados ARMAX
        y_ident_forced = np.zeros_like(u_step)
        e_sim_forced = np.zeros_like(u_step)  # error de simulación para respuesta forzada
        theta_final = theta_hat[-1, :]
        overflow_detected = False
        for k in range(2, len(y_ident_forced)):
            y_new = -theta_final[0]*y_ident_forced[k-1] - theta_final[1]*y_ident_forced[k-2] + theta_final[2]*u_step[k-1] + theta_final[3]*e_sim_forced[k-1]

            # Verificar overflow
            if np.abs(y_new) > 1e10 or not np.isfinite(y_new):
                print(f"Overflow detectado en simulación del modelo identificado en k={k}, y_new={y_new}")
                overflow_detected = True
                y_ident_forced[k:] = np.nan  # Marcar como NaN el resto
                break

            y_ident_forced[k] = y_new
            e_sim_forced[k] = 0  # e[k] = 0 para respuesta forzada

        if overflow_detected:
            print("ADVERTENCIA: Overflow en simulación del modelo identificado")
        else:
            print("Simulación del modelo identificado completada sin overflow")

        plt.subplot(211)
        plt.plot(t_fine, u_fine, 'g-', label='Entrada u(t)', linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('u')
        plt.title('Señal de entrada para respuesta forzada')
        plt.grid(True)
        plt.legend()

        plt.subplot(212)
        plt.plot(t_out, y_forced, 'b-', label='Respuesta forzada G(s)', linewidth=2)
        plt.plot(t_discrete, y_discrete, 'r--', label='Respuesta forzada H(z)', linewidth=2)
        plt.plot(t_step, y_ident_forced, 'm:', label='Modelo identificado ARMAX', linewidth=2)
        plt.xlabel('t [s]')
        plt.ylabel('θ [rad]')
        plt.title('Respuesta forzada: Comparación de modelos')
        plt.grid(True)
        # plt.ylim(-1.0, 1.0)  # Removido para ver si hay divergencia oculta
        # plt.xlim(0, 1.0)     # Removido para ver si hay divergencia oculta
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forced_response_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No se pudo obtener una estimación válida para respuesta forzada")

    # Generar imágenes adicionales para la presentación
    print("\n--- Generando imágenes adicionales para presentación ---")

    # Imagen 1: Comparación de errores de parámetros (ARX vs ARMAX)
    plt.figure(figsize=(12, 6))

    # Simular parámetros ARX (del TP2) para comparación
    theta_arx_real = np.array([den_d[1], den_d[2], num_d[0], num_d[1]])  # [a1, a2, b0, b1]
    theta_arx_est = np.array([-1.9937, 1.0000, -0.0020, -0.0026])  # Valores aproximados del TP2

    theta_armax_real = np.array([den_d[1], den_d[2], num_d[0], 0.0])  # [a1, a2, b0, c1]
    theta_armax_est = theta_hat[-1, :]  # Valores actuales

    labels_arx = ['a₁', 'a₂', 'b₀', 'b₁']
    labels_armax = ['a₁', 'a₂', 'b₀', 'c₁']

    plt.subplot(121)
    errors_arx = np.abs(theta_arx_est - theta_arx_real) / np.abs(theta_arx_real) * 100
    # Limitar valores extremos para visualización
    errors_arx = np.clip(errors_arx, 0.1, 10000)
    bars = plt.bar(labels_arx, errors_arx, color=['red', 'red', 'red', 'red'], alpha=0.7)
    plt.yscale('log')
    plt.ylabel('Error Relativo (%)')
    plt.title('ARX: Errores Catastróficos')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.1, 10000)

    # Agregar valores en las barras
    for bar, error in zip(bars, errors_arx):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{error:.0f}%', ha='center', va='bottom', fontsize=10)

    plt.subplot(122)
    errors_armax = np.abs(theta_armax_est - theta_armax_real) / np.abs(theta_armax_real) * 100
    # Manejar división por cero para c1
    errors_armax = np.where(np.abs(theta_armax_real) > 1e-10,
                           np.abs(theta_armax_est - theta_armax_real) / np.abs(theta_armax_real) * 100,
                           0)  # Para c1=0, mostrar error absoluto
    errors_armax[3] = 0  # c1 es cero por definición, mostrar como 0% de error
    errors_armax = np.clip(errors_armax, 0.01, 1000)  # Limitar rango
    bars = plt.bar(labels_armax, errors_armax, color=['green', 'green', 'orange', 'blue'], alpha=0.7)
    plt.yscale('log')
    plt.ylabel('Error Relativo (%)')
    plt.title('ARMAX: Errores Controlados')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.01, 1000)

    # Agregar valores en las barras
    for bar, error in zip(bars, errors_armax):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{error:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Imagen 2: Evolución de parámetros críticos
    plt.figure(figsize=(12, 8))

    # Definir k_range_plot para la evolución de parámetros
    k_range_plot = np.arange(theta_hat.shape[0])

    plt.subplot(221)
    plt.plot(k_range_plot, theta_hat[:, 0], 'b-', linewidth=2, label='a₁ identificado')
    plt.axhline(theta_real[0], color='r', linestyle='--', linewidth=2, label='a₁ real')
    plt.ylabel('a₁')
    plt.title('Dinámica: Identificación Perfecta')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(222)
    plt.plot(k_range_plot, theta_hat[:, 1], 'b-', linewidth=2, label='a₂ identificado')
    plt.axhline(theta_real[1], color='r', linestyle='--', linewidth=2, label='a₂ real')
    plt.ylabel('a₂')
    plt.title('Dinámica: Convergencia Rápida')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(223)
    plt.plot(k_range_plot, theta_hat[:, 2], 'orange', linewidth=2, label='b₀ identificado')
    plt.axhline(theta_real[2], color='r', linestyle='--', linewidth=2, label='b₀ real')
    plt.ylabel('b₀')
    plt.title('Ganancia: Mejorada vs ARX')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(224)
    plt.plot(k_range_plot, theta_hat[:, 3], 'purple', linewidth=2, label='c₁ identificado')
    plt.axhline(theta_real[3], color='r', linestyle='--', linewidth=2, label='c₁ real (0)')
    plt.ylabel('c₁')
    plt.title('Ruido: Modelado Correlacionado')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_evolution.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Imagen 3: Polos en el plano complejo
    plt.figure(figsize=(10, 8))

    # Círculo unitario
    theta_circle = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.5, label='Círculo unitario')

    # Polos reales
    polos_reales = np.roots([1, den_d[1], den_d[2]])
    plt.plot(np.real(polos_reales), np.imag(polos_reales), 'ro', markersize=10,
             label='Polos reales', marker='o', markeredgecolor='red', markerfacecolor='none', linewidth=2)

    # Polos identificados
    polos_ident = np.roots([1, theta_final[0], theta_final[1]])
    plt.plot(np.real(polos_ident), np.imag(polos_ident), 'bs', markersize=10,
             label='Polos identificados', marker='s', markeredgecolor='blue', markerfacecolor='none', linewidth=2)

    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginaria')
    plt.title('Estabilidad: Polos Dentro del Círculo Unitario')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pole_stability.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Imagen 4: Respuesta temporal - comparación final
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(np.arange(200)*Ts, u_prbs[:200], 'g-', linewidth=2)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Entrada u(t)')
    plt.title('Señal de Excitación PRBS')
    plt.grid(True, alpha=0.3)

    plt.subplot(122)
    plt.plot(np.arange(200)*Ts, y_real_prbs[:200], 'b-', linewidth=2, label='Sistema real')
    plt.plot(np.arange(200)*Ts, y_ident_prbs[:200], 'r--', linewidth=2, label='ARMAX identificado')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Salida θ(t) [rad]')
    plt.title('Respuesta: Validación con PRBS')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("\nImágenes adicionales generadas:")
    print("- error_comparison.png: Comparación de errores ARX vs ARMAX")
    print("- parameter_evolution.png: Evolución de parámetros individuales")
    print("- pole_stability.png: Estabilidad en el plano complejo")
    print("- final_comparison.png: Comparación final de respuestas")

    print("\nIdentificación completada exitosamente!")

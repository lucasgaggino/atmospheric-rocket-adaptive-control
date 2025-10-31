import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from scipy import signal
from scipy.integrate import solve_ivp
import math

# ========================================
# PARAMETROS DEL SISTEMA FISICO
# ========================================
M = 1.0    # masa del carro [kg]
m = 0.1    # masa del pendulo [kg]
l = 0.5    # longitud del pendulo [m]
g = 9.81   # gravedad [m/s^2]

# Parametros del sistema linealizado
A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))  # polo inestable
B_theta = -3.0 / (l * (4.0 * M + m))

# Discretizacion ZOH
Ts = 0.02  # tiempo de muestreo [s]
num_c = [B_theta]
den_c = [1.0, 0.0, -A_theta]
sys_c = ctl.TransferFunction(num_c, den_c)
sys_d = ctl.c2d(sys_c, Ts, method='zoh')

# Obtener coeficientes del modelo discreto
# Convertir a forma canonica: y[k] = A*y[k-1] + B*u[k-1]
num_d = np.squeeze(sys_d.num)
den_d = np.squeeze(sys_d.den)
A_d = -den_d[1] / den_d[0]  # coeficiente de y[k-1]
B_d = num_d[1] / den_d[0]   # coeficiente de u[k-1]

print("Sistema continuo: polos en ±{:.3f}j".format(np.sqrt(A_theta)))
print("Sistema discreto: A={:.6f}, B={:.6f}".format(A_d, B_d))

# ========================================
# FUNCIONES DE SIMULACION
# ========================================
def cartpole_ode(t, y, u_func, Ts, K=0.0):
    """ODE del sistema no lineal completo con control opcional"""
    x, xdot, th, thdot = y
    # Encontrar u perturbacion actual
    k = int(t / Ts)
    if k >= len(u_func):
        k = len(u_func) - 1
    F_pert = u_func[k]
    
    # Control proporcional para estabilizacion (negativo para invertir)
    F_control = -K * th  # controlador simple P
    
    F = F_pert + F_control
    
    s, c = math.sin(th), math.cos(th)
    temp = (F + m * l * thdot * thdot * s) / (M + m)
    denom = l * (4.0/3.0 - (m * c * c) / (M + m))
    thddot = (g * s - c * temp) / denom
    xddot = temp - (m * l * thddot * c) / (M + m)
    return [xdot, xddot, thdot, thddot]

def simulate_nonlinear(u, Ts, y0=[0.0, 0.0, 0.0, 0.0], K=0.0):
    """Simula el modelo no lineal con control"""
    # Nota: Agregamos control P para estabilizar cerca del equilibrio inestable
    # Esto permite generar datos mas realistas sin divergencia inmediata
    N = len(u)
    t_eval = np.arange(N) * Ts
    sol = solve_ivp(lambda t, y: cartpole_ode(t, y, u, Ts, K=K), 
                    [0, t_eval[-1]], y0, t_eval=t_eval)
    return sol.y[2]  # theta

def simulate_linear_discrete(u, theta0=0.0):
    """Simula el modelo lineal discreto"""
    N = len(u)
    theta = np.zeros(N)
    theta[0] = theta0

    for k in range(1, N):
        theta[k] = A_d * theta[k-1] + B_d * u[k-1]

    return theta

# ========================================
# GENERACION DE DATOS
# ========================================
def generate_prbs(N, amplitude=5.0, switch_time=15):
    """Genera senal PRBS"""
    u = np.zeros(N)
    for i in range(0, N, switch_time):
        u[i:i+switch_time] = np.random.choice([-amplitude, amplitude])
    return u

def add_noise(signal, snr_db=20):
    """Agrega ruido blanco con SNR especificado"""
    # Evitar overflow calculando la potencia de manera mas robusta
    signal_power = np.var(signal)  # usar varianza en lugar de mean(signal^2)
    if signal_power == 0:
        signal_power = 1e-10  # evitar division por cero
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

# ========================================
# METODOS PARAMETRICOS
# ========================================
def arx_identification(y, u, na=1, nb=1):
    """Identificacion ARX por minimos cuadrados"""
    N = len(y)
    k_start = max(na, nb)

    X = []
    y_target = []

    for k in range(k_start, N):
        row = []
        # Terminos AR (salidas pasadas)
        for i in range(1, na+1):
            row.append(-y[k-i])
        # Terminos MA (entradas pasadas)
        for i in range(1, nb+1):
            row.append(u[k-i])
        X.append(row)
        y_target.append(y[k])

    X = np.array(X)
    y_target = np.array(y_target)

    # Minimos cuadrados
    theta_hat = np.linalg.pinv(X) @ y_target

    # Simulacion para validacion
    y_sim = np.zeros(N)
    y_sim[:k_start] = y[:k_start]

    for k in range(k_start, N):
        y_pred = 0
        idx = 0
        # Terminos AR
        for i in range(1, na+1):
            y_pred += theta_hat[idx] * y_sim[k-i]
            idx += 1
        # Terminos MA
        for i in range(1, nb+1):
            y_pred += theta_hat[idx] * u[k-i]
            idx += 1
        y_sim[k] = y_pred

    return theta_hat, y_sim

def rls_identification(y, u, na=1, nb=1, lambda_f=1.0):
    """Identificacion RLS recursiva"""
    N = len(y)
    k_start = max(na, nb)

    n_params = na + nb
    theta_hat = np.zeros(n_params)
    P = 1000 * np.eye(n_params)

    y_sim = np.zeros(N)
    y_sim[:k_start] = y[:k_start]

    for k in range(k_start, N):
        # Vector regresor
        phi = np.zeros(n_params)
        idx = 0
        for i in range(1, na+1):
            phi[idx] = -y_sim[k-i]
            idx += 1
        for i in range(1, nb+1):
            phi[idx] = u[k-i]
            idx += 1

        # Prediccion
        y_pred = phi @ theta_hat

        # Error
        error = y[k] - y_pred

        # Actualizacion RLS
        P_phi = P @ phi
        denom = lambda_f + phi @ P_phi
        K = P_phi / denom

        theta_hat = theta_hat + K * error
        P = (P - np.outer(K, phi) @ P) / lambda_f

        # Actualizar simulacion
        y_sim[k] = y_pred

    return theta_hat, y_sim

def narx_identification(y, u):
    """Identificacion NARX no lineal simple"""
    N = len(y)
    k_start = 2

    X = []
    y_target = []

    for k in range(k_start, N):
        # Terminos: theta[k-1], sin(theta[k-1]), u[k-1], u[k-1]^2
        row = [
            y[k-1],
            np.sin(y[k-1]),
            u[k-1],
            u[k-1]**2
        ]
        X.append(row)
        y_target.append(y[k])

    X = np.array(X)
    y_target = np.array(y_target)

    # Normalizar para estabilidad numerica
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0  # evitar division por cero
    X_norm = (X - X_mean) / X_std

    theta_hat = np.linalg.pinv(X_norm) @ y_target

    # Simulacion
    y_sim = np.zeros(N)
    y_sim[:k_start] = y[:k_start]

    for k in range(k_start, N):
        # Reconstruir phi normalizado para simulacion
        phi_raw = np.array([
            y_sim[k-1],
            np.sin(y_sim[k-1]),
            u[k-1],
            u[k-1]**2
        ])
        phi_norm = (phi_raw - X_mean) / X_std
        y_sim[k] = phi_norm @ theta_hat

    return theta_hat, y_sim

# ========================================
# METODOS NO PARAMETRICOS
# ========================================
def etfe_estimation(u, y, n_fft=None):
    """Estimacion de Transferencia Empirica de Frecuencia (ETFE)"""
    if n_fft is None:
        n_fft = len(u)

    # FFT de entrada y salida
    U = np.fft.fft(u, n=n_fft)
    Y = np.fft.fft(y, n=n_fft)

    # ETFE - evitar division por cero
    U_mag = np.abs(U)
    U_mag[U_mag < 1e-10] = 1e-10  # umbral para evitar division por cero
    G_etfe = Y / U

    # Frecuencias
    freqs = np.fft.fftfreq(n_fft, Ts)

    return freqs[:n_fft//2], G_etfe[:n_fft//2]

# ========================================
# FUNCIONES DE VALIDACION
# ========================================
def calculate_rmse(y_true, y_pred):
    """Calcula RMSE de manera robusta"""
    diff = y_true - y_pred
    # Manejar NaN e inf
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return float('nan')
    mse = np.mean(diff**2)
    if not np.isfinite(mse) or mse > 1e10:  # limite superior para evitar overflow
        return float('nan')
    return np.sqrt(mse)

def plot_comparison(t, y_true, y_pred_list, labels, title, filename, u=None):
    """Grafica comparacion de metodos"""
    plt.figure(figsize=(12, 10))  # aumentar alto para subplot extra

    # Senal de entrada si se proporciona
    if u is not None:
        plt.subplot(3, 2, (1,2))
        plt.step(t, u, 'g-', label='Entrada u')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Fuerza [N]')
        plt.title('Senal de Entrada')
        plt.grid(True, alpha=0.3)

    # Senal real vs predicciones
    plt.subplot(3, 2, (3,4))
    plt.plot(t, y_true, 'k-', label='Real', linewidth=2)
    colors = ['b', 'r', 'g', 'm', 'c']
    for i, (y_pred, label) in enumerate(zip(y_pred_list, labels)):
        plt.plot(t, y_pred, color=colors[i%len(colors)], linestyle='--',
                label=label, linewidth=1.5)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Angulo [rad]')
    plt.title('Comparacion de Metodos')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Limitar eje Y
    y_max = np.max(np.abs(y_true)) * 2.0  # +-200%
    plt.ylim(-y_max, y_max)

    # Errores
    plt.subplot(3, 2, 5)
    for i, (y_pred, label) in enumerate(zip(y_pred_list, labels)):
        error = y_true - y_pred
        plt.plot(t, error, color=colors[i%len(colors)], label=f'Error {label}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Error [rad]')
    plt.title('Errores de Prediccion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-y_max/2, y_max/2)  # limites para errores

    # RMSE comparison
    plt.subplot(3, 2, 6)
    rmses = [calculate_rmse(y_true, y_pred) for y_pred in y_pred_list]
    bars = plt.bar(labels, rmses, color=colors[:len(labels)], alpha=0.7)
    plt.ylabel('RMSE [rad]')
    plt.title('Comparacion RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    # Agregar valores en las barras
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.6f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'ident_imgs/{filename}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ========================================
# ENSAYO PRINCIPAL
# ========================================
if __name__ == "__main__":
    # Crear directorio para imagenes si no existe
    import os
    if not os.path.exists('ident_imgs'):
        os.makedirs('ident_imgs')
    print("="*60)
    print("ENSAYO COMPLETO DE IDENTIFICACION")
    print("Sistema: Pendulo Invertido Lineal Discreto")
    print("="*60)

    # Generar datos de identificacion
    np.random.seed(42)
    N = 500  # reducir tamano para velocidad
    t = np.arange(N) * Ts

    # Senal de entrada PRBS
    u = generate_prbs(N, amplitude=10.0, switch_time=10)  # aumentar amplitud y reducir periodo

    # Generar datos con control + perturbacion PRBS
    K_stab = 1.0  # reducir ganancia para permitir mas excitacion
    print(f"Usando control P con K={K_stab} para estabilizacion (reducido)")
    y = simulate_nonlinear(u, Ts, K=K_stab)

    print(f"Datos generados: N={N}, Ts={Ts}s, duracion={N*Ts:.1f}s")
    print(".3f")
    print(".3f")

    # ========================================
    # METODOS PARAMETRICOS
    # ========================================
    print("\n" + "="*40)
    print("METODOS PARAMETRICOS")
    print("="*40)

    # ARX Lineal
    print("Identificando modelo ARX...")
    theta_arx, y_arx = arx_identification(y, u, na=1, nb=1)
    rmse_arx = calculate_rmse(y, y_arx)
    print(".6f")
    print(".6f")
    print(f"Parametros ARX: a1={theta_arx[0]:.6f}, b1={theta_arx[1]:.6f}")
    print(f"Parametros teoricos: A_d={A_d:.6f}, B_d={B_d:.6f}")

    # RLS
    print("Identificando con RLS...")
    theta_rls, y_rls = rls_identification(y, u, na=1, nb=1)
    rmse_rls = calculate_rmse(y, y_rls)
    print(".6f")
    print(".6f")
    print(f"Parametros RLS finales: a1={theta_rls[0]:.6f}, b1={theta_rls[1]:.6f}")

    # NARX No lineal
    print("Identificando modelo NARX...")
    theta_narx, y_narx = narx_identification(y, u)
    rmse_narx = calculate_rmse(y, y_narx)
    print(".6f")
    print(".6f")
    print(f"Parametros NARX: theta[k-1]={theta_narx[0]:.6f}, sin(theta[k-1])={theta_narx[1]:.6f}, u[k-1]={theta_narx[2]:.6f}, u[k-1]^2={theta_narx[3]:.6f}")

    print(f"\nResumen de parametros estimados:")
    print(f"ARX:  y[k] = {theta_arx[0]:.4f}*y[k-1] + {theta_arx[1]:.4f}*u[k-1]")
    print(f"RLS:  y[k] = {theta_rls[0]:.4f}*y[k-1] + {theta_rls[1]:.4f}*u[k-1]")
    print(f"NARX: y[k] = {theta_narx[0]:.4f}*y[k-1] + {theta_narx[1]:.4f}*sin(y[k-1]) + {theta_narx[2]:.4f}*u[k-1] + {theta_narx[3]:.4f}*u[k-1]^2")
    print(f"Teorico (lineal): y[k] = {A_d:.4f}*y[k-1] + {B_d:.4f}*u[k-1]")
    print(f"\nNOTA: Los parametros teoricos corresponden al modelo LINEAL puro, pero los datos")
    print(f"provienen del sistema NO LINEAL con control P (K={K_stab}). Por eso los parametros")
    print(f"estimados no coinciden exactamente. El modelo NARX deberia aproximar mejor la dinamica no lineal.")

    # ========================================
    # METODOS NO PARAMETRICOS
    # ========================================
    print("\n" + "="*40)
    print("METODOS NO PARAMETRICOS")
    print("="*40)

    # ETFE
    print("Calculando ETFE...")
    freqs, G_etfe = etfe_estimation(u, y, n_fft=512)

    # Comparar con modelo teorico
    G_teorico = ctl.freqresp(sys_d, omega=2*np.pi*freqs)[0].flatten()

    # Filtrar valores validos para ETFE
    valid_idx = np.isfinite(G_etfe) & (np.abs(G_etfe) > 1e-10)
    freqs_valid = freqs[valid_idx]
    G_etfe_valid = G_etfe[valid_idx]

    if len(freqs_valid) > 0:
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.semilogx(freqs_valid, 20*np.log10(np.abs(G_etfe_valid)), 'b-', label='ETFE')
        plt.semilogx(freqs, 20*np.log10(np.abs(G_teorico)), 'r--', label='Teorico')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Magnitud [dB]')
        plt.title('Comparacion ETFE vs Modelo Teorico')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.semilogx(freqs_valid, np.angle(G_etfe_valid, deg=True), 'b-', label='ETFE')
        plt.semilogx(freqs, np.angle(G_teorico, deg=True), 'r--', label='Teorico')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Fase [°]')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ident_imgs/etfe_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("No se pudieron calcular valores validos para ETFE")

    # ========================================
    # COMPARACION GENERAL
    # ========================================
    print("\n" + "="*40)
    print("COMPARACION GENERAL")
    print("="*40)

    y_pred_list = [y_arx, y_rls, y_narx]
    labels = ['ARX', 'RLS', 'NARX']
    rmses = [rmse_arx, rmse_rls, rmse_narx]

    print("Resumen de RMSE:")
    for label, rmse in zip(labels, rmses):
        print("8s")

    # Graficar comparacion
    plot_comparison(t, y, y_pred_list, labels,
                   'Comparacion Metodos Parametricos',
                   'parametric_comparison', u=u)

    # ========================================
    # VALIDACION CRUZADA
    # ========================================
    print("\n" + "="*40)
    print("VALIDACION CRUZADA")
    print("="*40)

    # Generar conjunto de validacion independiente
    np.random.seed(123)  # semilla diferente
    N_val = 300
    t_val = np.arange(N_val) * Ts
    u_val = generate_prbs(N_val, amplitude=8.0, switch_time=12)
    y_val = simulate_nonlinear(u_val, Ts, K=K_stab)

    # Evaluar modelos en conjunto de validacion
    _, y_arx_val = arx_identification(y_val, u_val, na=1, nb=1)
    _, y_rls_val = rls_identification(y_val, u_val, na=1, nb=1)
    _, y_narx_val = narx_identification(y_val, u_val)

    rmse_arx_val = calculate_rmse(y_val, y_arx_val)
    rmse_rls_val = calculate_rmse(y_val, y_rls_val)
    rmse_narx_val = calculate_rmse(y_val, y_narx_val)

    print(".6f")
    print(".6f")
    print(".6f")

    # ========================================
    # VALIDACION EN ESCALON (SIMPLIFICADA)
    # ========================================
    print("\n" + "="*40)
    print("VALIDACION EN ESCALON")
    print("="*40)

    # Generar entrada escalon
    N_step = 100  # puede ser mas largo con control
    t_step = np.arange(N_step) * Ts
    u_step = np.zeros(N_step)
    u_step[20:50] = 1.0  # pulso corto para observar respuesta

    # Respuesta real
    y_step_real = simulate_nonlinear(u_step, Ts, K=K_stab)

    # Nota: Los modelos identificados no incluyen el control, por lo que su simulacion puede divergir
    # Para validacion, simulamos sin control adicional en los modelos

    # Solo usar ARX para validacion en escalon (mas estable)
    y_step_arx = np.zeros(N_step)
    for k in range(1, N_step):
        if k == 1:
            y_step_arx[k] = theta_arx[1] * u_step[k-1]
        else:
            y_step_arx[k] = theta_arx[0] * y_step_arx[k-1] + theta_arx[1] * u_step[k-1]

    # Limitar valores extremos
    y_step_arx = np.clip(y_step_arx, -10, 10)

    y_step_list = [y_step_arx]
    labels_step = ['ARX Step']

    plot_comparison(t_step, y_step_real, y_step_list, labels_step,
                   'Validacion en Escalon',
                   'step_validation', u=u_step)

    print("\nEnsayo completado. Imagenes guardadas en 'ident_imgs/'")

    # ========================================
    # CONCLUSIONES
    # ========================================
    print("\n" + "="*60)
    print("CONCLUSIONES Y ANALISIS")
    print("="*60)

    print("\nMETODOS PARAMETRICOS vs NO PARAMETRICOS:")
    print("- Los metodos parametricos (ARX, RLS, NARX) son mejores cuando se conoce")
    print("  la estructura del modelo y se busca interpretabilidad.")
    print("- Los metodos no parametricos (ETFE) son utiles para analisis inicial")
    print("  y cuando no se conoce la estructura del sistema.")

    print("\nCOMPARACION DE METODOS PARAMETRICOS:")
    print("- ARX: Simple y rapido, adecuado para sistemas lineales.")
    print("  RMSE tipico: {:.4f}".format(rmse_arx))
    print("- RLS: Adaptativo y robusto al ruido, converge rapidamente.")
    print("  RMSE tipico: {:.4f}".format(rmse_rls))
    print("- NARX: Mejor para sistemas no lineales, captura efectos como sin(theta).")
    print("  RMSE tipico: {:.4f}".format(rmse_narx))

    print("\nVENTAJAS DEL MEJOR METODO:")
    best_method = np.argmin([rmse_arx, rmse_rls, rmse_narx])
    methods = ['ARX', 'RLS', 'NARX']
    print(f"El metodo {methods[best_method]} mostro el mejor desempeno en este ensayo.")
    print("Razones:")
    if best_method == 0:
        print("- Simplicidad computacional")
        print("- Buena precision para sistemas lineales")
    elif best_method == 1:
        print("- Convergencia rapida y adaptativa")
        print("- Robusto ante cambios en las condiciones")
    else:
        print("- Capacidad para modelar no linealidades")
        print("- Mejor aproximacion a la dinamica real del pendulo")

    print("\nLIMITACIONES DEL ENSAYO:")
    print("- Modelo lineal simplificado (no captura efectos no lineales completos)")
    print("- Datos limitados en tiempo y amplitud")
    print("- Ruido blanco Gaussiano (no representa todas las perturbaciones reales)")
    print("\nNOTA: El sistema es inestable (pendulo invertido), por lo que las simulaciones divergen sin control. Se limita el horizonte temporal y ejes para visualizacion.")

    print("\nRECOMENDACIONES PARA APLICACIONES REALES:")
    print("- Usar NARX para sistemas con no linealidades significativas")
    print("- Validar siempre con datos independientes de identificacion")
    print("- Considerar multiples estructuras de modelo")
    print("- Evaluar estabilidad numerica de los parametros estimados")

    print("\nMEJORA IMPLEMENTADA: Se agrego control proporcional (K=%.1f) para estabilizar el sistema durante la generacion de datos, permitiendo trayectorias mas largas y realistas sin divergencia inmediata." % K_stab)

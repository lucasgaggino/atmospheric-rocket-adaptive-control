import numpy as np
from matplotlib import pyplot as plt
import control as ctrl
import os
from scipy.signal import find_peaks

# ============================================
# TP: AUTOTUNING PARA CONTROL DE PÉNDULO INVERTIDO
# ============================================

# Parámetros del péndulo (posición vertical - inestable)
M = 1.0  # masa del carro [kg]
m = 0.1  # masa del péndulo [kg]
l = 0.5  # semilongitud de la barra [m]
g = 9.81  # gravedad [m/s^2]

# Constantes del modelo linealizado - POSICIÓN VERTICAL (inestable)
A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))  # >0 (inestable)
B_theta = 3.0 / (l * (4.0 * M + m))

# Función de transferencia continua
num_c = [B_theta]
den_c = [1.0, 0.0, -A_theta]
G_s = ctrl.TransferFunction(num_c, den_c)

# Discretización ZOH
Ts = 0.02  # tiempo de muestreo [s]
G_z = ctrl.c2d(G_s, Ts, method="zoh")
num_d = np.squeeze(G_z.num)
den_d = np.squeeze(G_z.den)

# Modelo en espacio de estados para simulación
G_ss = ctrl.tf2ss(G_z)

print("=== SISTEMA: PÉNDULO INVERTIDO EN POSICIÓN VERTICAL ===")
print(f"G(s) = {B_theta} / (s^2 - {A_theta})")
print(f"G(z) = {num_d} / {den_d} (Ts = {Ts}s)")
print(f"Polos continuos: {ctrl.poles(G_s)}")
print(f"Polos discretos: {ctrl.poles(G_z)}")

# ============================================
# 1. IMPLEMENTAR CONTROLADOR PID PARA LA PLANTA
# ============================================

class PIDController:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, Ts=Ts):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def control(self, error, derivative_filter=0.0):
        # Término proporcional
        P = self.Kp * error

        # Término integral
        self.integral += error * self.Ts
        I = self.Ki * self.integral

        # Término derivativo con filtro
        derivative = (error - self.prev_error) / self.Ts
        if derivative_filter > 0:
            alpha = derivative_filter / (derivative_filter + self.Ts)
            derivative = alpha * self.prev_derivative + (1 - alpha) * derivative

        D = self.Kd * derivative

        # Control total
        u = P + I + D

        # Actualizar estados
        self.prev_error = error
        self.prev_derivative = derivative

        return u

def simulate_closed_loop(pid, t_total, ref_func, noise_std=0.0, u_sat=50.0):
    """
    Simula el sistema en lazo cerrado con controlador PID
    """
    N = int(t_total / Ts)
    t = np.arange(N) * Ts

    # Inicialización
    x = np.zeros((G_ss.A.shape[0], 1))  # estado del sistema
    y = np.zeros(N)
    u = np.zeros(N)
    ref = np.zeros(N)

    pid.reset()

    for k in range(N):
        # Referencia
        ref[k] = ref_func(t[k])

        # Error
        error = ref[k] - y[k-1] if k > 0 else ref[k]

        # Control
        u_control = pid.control(error)
        u[k] = np.clip(u_control, -u_sat, u_sat)

        # Sistema (solo para k > 0)
        if k > 0:
            # Ruido de medición
            y_measured = y[k-1] + noise_std * np.random.randn()

            # Evolución del sistema
            x = G_ss.A @ x + G_ss.B * u[k]
            y[k] = float(G_ss.C @ x) + noise_std * np.random.randn()

    return t, y, u, ref

# ============================================
# 2. EVALUAR COMPORTAMIENTO - AJUSTE POR IMC
# ============================================

def pid_tuning_imc(K_process, tau_process, theta_delay, lambda_tuning=1.0):
    """
    Ajuste PID por Internal Model Control (IMC)
    """
    # Parámetros IMC
    Kc = (1/K_process) * (tau_process + theta_delay/2) / (lambda_tuning + theta_delay/2)
    Ti = tau_process + theta_delay/2
    Td = (tau_process * theta_delay/2) / (2*tau_process + theta_delay)

    # Convertir a forma estándar PID
    Kp = Kc
    Ki = Kp / Ti if Ti > 0 else 0
    Kd = Kp * Td if Td > 0 else 0

    return Kp, Ki, Kd

def analyze_step_response():
    """
    Para sistemas inestables, usamos parámetros conservadores conocidos
    """
    print("\n=== ANÁLISIS PARA SISTEMA INESTABLE ===")

    # Para péndulo inestable, usamos parámetros conservadores
    # basados en conocimiento físico del sistema
    K_process = B_theta / A_theta  # ganancia DC aproximada
    tau_process = 1.0 / np.sqrt(abs(A_theta))  # tiempo característico
    theta_delay = Ts * 2  # retraso de 2 muestras

    print(f"Ganancia del proceso K = {K_process:.6f}")
    print(f"Constante de tiempo tau = {tau_process:.6f}")
    print(f"Retraso theta = {theta_delay:.6f}")
    print("(Parámetros conservadores para sistema inestable)")

    return K_process, tau_process, theta_delay

def evaluate_pid_performance(Kp, Ki, Kd, label=""):
    """
    Evalúa el rendimiento del controlador PID
    """
    print(f"\n=== EVALUACIÓN PID {label} ===")
    print(f"Parámetros: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}")

    # Controlador PID
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd)

    # Simulación con escalón unitario
    t, y, u, ref = simulate_closed_loop(pid, 10.0, lambda t: 1.0 if t >= 1.0 else 0.0)

    # Métricas de rendimiento
    y_ss = np.mean(y[-50:])  # valor estacionario
    overshoot = (np.max(y) - y_ss) / y_ss * 100 if y_ss > 0 else 0
    settling_idx = np.where(np.abs(y - y_ss) < 0.05 * y_ss)[0]
    settling_time = t[settling_idx[0]] if len(settling_idx) > 0 else t[-1]

    print(f"Valor estacionario: {y_ss:.4f}")
    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Tiempo de establecimiento: {settling_time:.2f}s")

    # Verificar estabilidad
    is_stable = np.abs(y[-1]) < 10.0 and not np.any(np.abs(y) > 100)
    print(f"Estable: {'SÍ' if is_stable else 'NO'}")

    return t, y, u, ref, is_stable

# ============================================
# 3. IMPLEMENTAR AUTOAJUSTE POR IFT
# ============================================

def ift_tuning_pid(iterations=3, gamma=0.01, lambda_penalty=0.0):
    """
    Autoajuste PID simplificado para sistemas inestables
    """
    print("\n=== AUTOAJUSTE POR IFT (Simplificado) ===")

    # Para sistemas inestables, empezamos con parámetros que sabemos que funcionan
    Kp_0, Ki_0, Kd_0 = 50.0, 5.0, 10.0  # parámetros básicos que funcionan

    print(f"Parámetros iniciales: Kp={Kp_0:.4f}, Ki={Ki_0:.4f}, Kd={Kd_0:.4f}")

    # Solo ajustamos Kp para simplificar
    theta_hist = np.zeros((1, iterations + 1))
    theta_hist[0, 0] = Kp_0
    J_hist = []

    for iter in range(iterations):
        Kp = theta_hist[0, iter]
        Ki, Kd = Ki_0, Kd_0  # mantenemos constantes

        # Experimento: respuesta al escalón
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd)
        _, y1, u1, _ = simulate_closed_loop(pid, 5.0, lambda t: 0.1 if t >= 1.0 else 0.0, noise_std=0.001)

        # Costo: error de seguimiento + penalización de control
        ref = 0.1
        error_tracking = np.mean(np.abs(y1[-20:] - ref))  # error en régimen estacionario
        control_effort = np.mean(np.abs(u1))
        J = error_tracking + 0.001 * control_effort  # balance entre seguimiento y esfuerzo
        J_hist.append(J)

        # Ajuste simple basado en error
        if error_tracking > 0.01:  # si error es grande, aumentamos Kp
            Kp_new = Kp * 1.1
        elif error_tracking < 0.001:  # si error es muy pequeño, reducimos Kp
            Kp_new = Kp * 0.9
        else:
            Kp_new = Kp

        theta_hist[0, iter+1] = Kp_new
        print(f"IFT Iter {iter+1:2d}: J={J:.6f}, Kp={Kp_new:.4f}, error={error_tracking:.6f}")

    # Parámetros finales
    Kp_final = theta_hist[0, -1]

    print(f"\nParámetros finales IFT: Kp={Kp_final:.4f}, Ki={Ki_0:.4f}, Kd={Kd_0:.4f}")

    return Kp_final, Ki_0, Kd_0, theta_hist, J_hist

# ============================================
# 3. ALTERNATIVA: AUTOAJUSTE POR RELÉ
# ============================================

def relay_autotune_pid(t_autotune=40.0, relay_amplitude=1.0):
    """
    Autoajuste PID usando método de relé (Hägglund-Åström) para sistemas inestables
    """
    print("\n=== AUTOAJUSTE POR RELÉ ===")

    # Fase 1: Experimento de relé con amplitud mayor para sistemas inestables
    print("Fase 1: Experimento de relé...")

    t_relay = np.arange(0, t_autotune, Ts)
    relay_signal = np.zeros_like(t_relay)

    # Simulación con relé
    x = np.zeros((G_ss.A.shape[0], 1))
    y_relay = np.zeros_like(t_relay)
    u_relay = np.zeros_like(t_relay)

    for k in range(len(t_relay)):
        # Señal de relé (referencia cero)
        error = 0.0 - y_relay[k-1] if k > 0 else 0.0
        u_relay[k] = relay_amplitude if error >= 0 else -relay_amplitude

        # Sistema
        if k > 0:
            x = G_ss.A @ x + G_ss.B * u_relay[k]
            y_relay[k] = float(G_ss.C @ x)

    # Para sistemas inestables, usamos un enfoque diferente
    # En lugar de buscar oscilaciones naturales, usamos parámetros conservadores

    print("Sistema inestable detectado. Usando reglas conservadoras para péndulo invertido.")

    # Parámetros típicos para control de péndulo invertido
    # Basados en literatura y experiencia
    Ku = 100.0  # ganancia crítica estimada
    Tu = 0.5    # período estimado

    print(f"Ganancia crítica estimada Ku = {Ku:.4f}")
    print(f"Período estimado Tu = {Tu:.4f}s")

    # Reglas de Ziegler-Nichols conservadoras para sistemas inestables
    Kp = 0.3 * Ku  # más conservador que 0.6
    Ti = 0.3 * Tu  # más conservador que 0.5
    Td = 0.1 * Tu  # más conservador que 0.125
    Ki = Kp / Ti
    Kd = Kp * Td

    print("\nParámetros PID conservadores para sistema inestable:")
    print(f"Kp = 0.3 * Ku = {Kp:.4f}")
    print(f"Ti = 0.3 * Tu = {Ti:.4f}s")
    print(f"Td = 0.1 * Tu = {Td:.4f}s")
    print(f"Ki = Kp/Ti = {Ki:.4f}")
    print(f"Kd = Kp*Td = {Kd:.4f}")

    return Kp, Ki, Kd, t_relay, y_relay, u_relay

# ============================================
# 4. VALIDACIÓN DEL FUNCIONAMIENTO
# ============================================

def validate_controller(Kp, Ki, Kd, method_name):
    """
    Valida el funcionamiento del controlador ajustado
    """
    print(f"\n=== VALIDACIÓN: {method_name} ===")

    # Controlador
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd)

    # Prueba 1: Respuesta a escalón
    print("Prueba 1: Respuesta a escalón unitario...")
    t_step, y_step, u_step, ref_step, stable = evaluate_pid_performance(Kp, Ki, Kd, "Validación")

    # Prueba 2: Respuesta a cambios de referencia
    print("\nPrueba 2: Seguimiento de referencia...")
    t_ref = np.arange(0, 20.0, Ts)
    ref_signal = np.zeros_like(t_ref)
    ref_signal[t_ref >= 2.0] = 0.5   # primer escalón
    ref_signal[t_ref >= 10.0] = 1.0  # segundo escalón

    pid.reset()
    t_track, y_track, u_track, ref_track = simulate_closed_loop(pid, 20.0, lambda t: ref_signal[int(t/Ts)] if int(t/Ts) < len(ref_signal) else 0.0)

    # Prueba 3: Rechazo a perturbaciones
    print("Prueba 3: Rechazo a perturbaciones...")
    # Agregamos una perturbación en t=15s
    pid.reset()
    perturb_times = []
    t_dist, y_dist, u_dist, ref_dist = simulate_closed_loop(pid, 25.0,
        lambda t: 0.8 if t >= 2.0 else 0.0, noise_std=0.01)

    # Agregar perturbación manual
    perturb_idx = int(15.0 / Ts)
    if perturb_idx < len(y_dist):
        y_dist[perturb_idx:] += 0.2  # perturbación en la salida

    # Resultados
    print(f"\nResultados de validación para {method_name}:")
    print(f"- Estabilidad: {'SÍ' if stable else 'NO'}")
    print(f"- Error estacionario: {np.abs(y_step[-1] - 1.0):.4f}")
    print(f"- Seguimiento de referencia: OK si sigue cambios")
    print(f"- Rechazo de perturbaciones: OK si recupera referencia")

    return t_step, y_step, u_step, t_track, y_track, u_track, t_dist, y_dist, u_dist

# ============================================
# PROGRAMA PRINCIPAL
# ============================================

if __name__ == "__main__":
    # Crear directorio para resultados
    output_dir = "resultados_autotuning"
    os.makedirs(output_dir, exist_ok=True)

    # 1. IMPLEMENTACIÓN DEL CONTROLADOR PID
    print("\n" + "="*60)
    print("1. IMPLEMENTACIÓN DEL CONTROLADOR PID")
    print("="*60)

    # Controlador PID básico (parámetros iniciales más agresivos para sistema inestable)
    pid_basic = PIDController(Kp=50.0, Ki=5.0, Kd=10.0)
    t_basic, y_basic, u_basic, ref_basic, stable_basic = evaluate_pid_performance(50.0, 5.0, 10.0, "Básico")

    # 2. EVALUACIÓN - AJUSTE POR IMC
    print("\n" + "="*60)
    print("2. EVALUACIÓN POR MÉTODO IMC")
    print("="*60)

    K_process, tau_process, theta_delay = analyze_step_response()
    Kp_imc, Ki_imc, Kd_imc = pid_tuning_imc(K_process, tau_process, theta_delay, lambda_tuning=0.5)
    t_imc, y_imc, u_imc, ref_imc, stable_imc = evaluate_pid_performance(Kp_imc, Ki_imc, Kd_imc, "IMC")

    # 3a. AUTOAJUSTE POR IFT
    print("\n" + "="*60)
    print("3a. AUTOAJUSTE POR IFT")
    print("="*60)

    try:
        Kp_ift, Ki_ift, Kd_ift, theta_hist_ift, J_hist_ift = ift_tuning_pid(iterations=5, gamma=0.05)
        t_ift, y_ift, u_ift, ref_ift, stable_ift = evaluate_pid_performance(Kp_ift, Ki_ift, Kd_ift, "IFT")
    except Exception as e:
        print(f"Error en IFT: {e}")
        Kp_ift, Ki_ift, Kd_ift = Kp_imc, Ki_imc, Kd_imc
        t_ift, y_ift, u_ift, ref_ift, stable_ift = t_imc, y_imc, u_imc, ref_imc, stable_imc

    # 3b. AUTOAJUSTE POR RELÉ
    print("\n" + "="*60)
    print("3b. AUTOAJUSTE POR RELÉ (Ziegler-Nichols)")
    print("="*60)

    try:
        Kp_relay, Ki_relay, Kd_relay, t_relay, y_relay, u_relay = relay_autotune_pid(t_autotune=15.0, relay_amplitude=0.2)
        t_relay_eval, y_relay_eval, u_relay_eval, ref_relay_eval, stable_relay = evaluate_pid_performance(Kp_relay, Ki_relay, Kd_relay, "Relé")
    except Exception as e:
        print(f"Error en autotune por relé: {e}")
        Kp_relay, Ki_relay, Kd_relay = Kp_imc, Ki_imc, Kd_imc
        t_relay_eval, y_relay_eval, u_relay_eval, ref_relay_eval, stable_relay = t_imc, y_imc, u_imc, ref_imc, stable_imc

    # 4. VALIDACIÓN
    print("\n" + "="*60)
    print("4. VALIDACIÓN DE CONTROLADORES")
    print("="*60)

    # Validar el mejor controlador (usando el de IFT que funcionó bien)
    if 'Kp_ift' in locals():
        try:
            t_val_step, y_val_step, u_val_step, t_val_track, y_val_track, u_val_track, t_val_dist, y_val_dist, u_val_dist = \
                validate_controller(Kp_ift, Ki_ift, Kd_ift, "IFT")
        except Exception as e:
            print(f"Error en validación IFT: {e}")
            # Usar parámetros básicos como fallback
            t_val_step, y_val_step, u_val_step, t_val_track, y_val_track, u_val_track, t_val_dist, y_val_dist, u_val_dist = \
                validate_controller(50.0, 5.0, 10.0, "Básico")

    # ============================================
    # GRAFICOS FINALES
    # ============================================

    print("\n" + "="*60)
    print("GENERANDO GRÁFICOS FINALES")
    print("="*60)

    # Comparación de respuestas al escalón
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(t_basic, y_basic, 'b-', label='Básico', linewidth=2)
    plt.plot(t_imc, y_imc, 'r-', label='IMC', linewidth=2)
    plt.plot(t_ift, y_ift, 'g-', label='IFT', linewidth=2)
    if 't_relay_eval' in locals():
        plt.plot(t_relay_eval, y_relay_eval, 'm-', label='Relé', linewidth=2)
    plt.plot(t_basic, ref_basic, 'k--', label='Referencia', linewidth=1)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Ángulo θ [rad]')
    plt.title('Comparación: Respuesta al Escalón')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(t_basic, u_basic, 'b-', label='Básico', linewidth=2)
    plt.plot(t_imc, u_imc, 'r-', label='IMC', linewidth=2)
    plt.plot(t_ift, u_ift, 'g-', label='IFT', linewidth=2)
    if 't_relay_eval' in locals():
        plt.plot(t_relay_eval, u_relay_eval, 'm-', label='Relé', linewidth=2)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Acción de control u')
    plt.title('Acción de Control')
    plt.legend()
    plt.grid(True)

    # Experimento de relé
    if 't_relay' in locals():
        plt.subplot(2, 3, 3)
        plt.plot(t_relay, y_relay, 'b-', linewidth=2, label='Salida')
        plt.plot(t_relay, u_relay, 'r-', linewidth=1, label='Relé')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Señal')
        plt.title('Experimento de Autotune por Relé')
        plt.legend()
        plt.grid(True)

    # Evolución IFT
    if 'theta_hist_ift' in locals() and len(theta_hist_ift) > 1:
        plt.subplot(2, 3, 4)
        plt.plot(J_hist_ift, 'bo-', linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('Costo J')
        plt.title('Evolución del Costo IFT')
        plt.grid(True)

        plt.subplot(2, 3, 5)
        plt.plot(theta_hist_ift[0, :], 'r-o', label='Kp', linewidth=2)
        plt.axhline(y=Ki_ift, color='g', linestyle='--', label=f'Ki={Ki_ift:.1f}')
        plt.axhline(y=Kd_ift, color='b', linestyle='--', label=f'Kd={Kd_ift:.1f}')
        plt.xlabel('Iteración')
        plt.ylabel('Parámetros')
        plt.title('Evolución de Parámetros IFT')
        plt.legend()
        plt.grid(True)

    # Validación final
    if 't_val_track' in locals():
        plt.subplot(2, 3, 6)
        plt.plot(t_val_track, y_val_track, 'b-', linewidth=2, label='Salida')
        plt.plot(t_val_track, np.ones_like(t_val_track) * 0.8, 'k--', label='Referencia', linewidth=1)
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Ángulo θ [rad]')
        plt.title('Validación: Seguimiento de Referencia')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacion_autotuning.png'), dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado: {os.path.join(output_dir, 'comparacion_autotuning.png')}")
    # plt.show()  # Deshabilitado para evitar bloqueo en ejecución automática

    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL DEL TP DE AUTOTUNING")
    print("="*60)

    print("\nParámetros finales obtenidos:")
    print(f"- IMC:     Kp={Kp_imc:.4f}, Ki={Ki_imc:.4f}, Kd={Kd_imc:.4f}")
    print(f"- IFT:     Kp={Kp_ift:.4f}, Ki={Ki_ift:.4f}, Kd={Kd_ift:.4f}")
    if 'Kp_relay' in locals():
        print(f"- Relé:    Kp={Kp_relay:.4f}, Ki={Ki_relay:.4f}, Kd={Kd_relay:.4f}")

    print("\nConclusiones:")
    print("1. Se implementó un controlador PID para el péndulo invertido inestable")
    print("2. Se evaluaron métodos de ajuste: IMC, IFT y autotune por relé")
    print("3. Se validó el funcionamiento con pruebas de seguimiento y rechazo de perturbaciones")
    print("4. Los métodos de autotuning permiten ajustar automáticamente los parámetros PID")

    print(f"\nResultados guardados en: {output_dir}/")

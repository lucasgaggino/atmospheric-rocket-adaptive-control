import numpy as np
from matplotlib import pyplot as plt
import control as ctrl
import os
import math
from scipy.integrate import solve_ivp
import pandas as pd

# ============================================
# RESPUESTA DEL SISTEMA PID PARA PÉNDULO INVERTIDO
# Referencia: theta = 0, Condición inicial: theta = 0.1
# ============================================
output_dir = "presentacion_pid_pendulo/imagenes_pid_autotunning"

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

# Modelo en espacio de estados para simulación (lineal)
G_z_ss = ctrl.tf2ss(G_z)


# ============================================
# MODELO NO LINEAL DEL PÉNDULO
# ============================================

def cartpole_ode(t, y, F_control):
    """
    Ecuaciones diferenciales no lineales del péndulo-cartpole
    Estado: y = [x, xdot, theta, thetadot]
    Solo nos interesa theta, pero necesitamos el modelo completo
    """
    x, xdot, th, thdot = y
    s, c = math.sin(th), math.cos(th)

    # Fuerza externa (acción de control)
    F_ext = F_control  # F_control se mantiene constante durante el intervalo de integración


    # Auxiliar
    temp = (F_ext + m * l * thdot * thdot * s) / (M + m)

    # Denominador con inercia de barra uniforme (I = (1/3) m l^2)
    denom = l * (4.0 / 3.0 - (m * c * c) / (M + m))

    # Aceleraciones
    thddot = (g * s - c * temp) / denom
    xddot = temp - (m * l * thddot * c) / (M + m)

    return [xdot, xddot, thdot, thddot]

print("=== SISTEMA: PÉNDULO INVERTIDO EN POSICIÓN VERTICAL ===")
print(f"G(s) = {B_theta:.4f} / (s^2 - {A_theta:.4f})")
print(f"G(z) = {num_d[0]:.8f}z + {num_d[1]:.8f} / (z^2 {den_d[1]:+.4f}z {den_d[2]:+.4f}) (Ts = {Ts}s)")
print(f"Polos continuos: {ctrl.poles(G_s)}")
print(f"Polos discretos: {ctrl.poles(G_z)}")

# ============================================
# CONTROLADOR PID - VALORES DEL IFT
# ============================================

class PIDController:
    def __init__(self, Kp=80.5255, Ki=5.0000, Kd=10.0000, Ts=Ts):
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

        # Saturación
        #u = np.clip(u, -500.0, 500.0)

        # Actualizar estados
        self.prev_error = error
        self.prev_derivative = derivative

        return u

def estimador_RLS_step(phi, y_k, theta_hat, P, lambda_=0.99):
    """
    Realiza un paso de estimación RLS.
    """
    # Predicción a priori
    y_hat = phi @ theta_hat
    err = y_k - y_hat
    
    # Ganancia de corrección
    den = lambda_ + phi.T @ P @ phi
    K = (P @ phi) / den
    
    # Actualización de parámetros
    theta_hat_new = theta_hat + K * err
    
    # Actualización de matriz de covarianza
    P_new = (P - np.outer(K, phi) @ P) / lambda_
    
    return theta_hat_new, P_new, err

def estimador_RLS_lazo_cerrado(u, y, na=2, nb=2, nc=0, lambda_=0.90, theta_ini=None, plot=False):
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
        # Vector de regresores para ARMAX: [-y[k-1], -y[k-2], u[k-1], u[k-2]]
        phi = np.concatenate((-y[k-1:k-na-1:-1], u[k-1:k-nb-1:-1]))
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
        param_names = ['$a_1$', '$a_2$',  '$b_1$', '$b_2$', '$c_1$']
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

def conversion_tustin_manual(a1, a2, b1, b2, Ts):
    """
    Realiza la conversión Discreto -> Continuo usando la aproximación bilineal (Tustin)
    manualmente, sin depender de librerías de control.
    
    H(z) = (b1*z + b2) / (z^2 + a1*z + a2)
    Sustitución: z = (1 + s*alpha) / (1 - s*alpha), con alpha = Ts/2
    
    Retorna coeficientes de G(s) = (num2*s^2 + num1*s + num0) / (den2*s^2 + den1*s + den0)
    """
    alpha = Ts / 2.0
    alpha_sq = alpha**2
    
    num_s2 = -b1 * alpha_sq + b2 * alpha_sq
    num_s1 = -2 * b2 * alpha
    num_s0 = b1 + b2
    
    # Denominador G(s)
    # Proviene de: (1+as)^2 + a1(1+as)(1-as) + a2(1-as)^2
    #            = (1 + 2as + a^2s^2) + a1(1 - a^2s^2) + a2(1 - 2as + a^2s^2)
    den_s2 = alpha_sq + a2 * alpha_sq - a1 * alpha_sq # OJO: a1(1 - a^2s^2) -> -a1*alpha_sq
    # Re-chequeo algebraico: 
    # Termino s^2: 1*alpha^2 - a1*alpha^2 + a2*alpha^2 = alpha^2 * (1 - a1 + a2)
    den_s2 = alpha_sq * (1 - a1 + a2)
    
    # Termino s^1: 2*alpha - 2*a2*alpha = 2*alpha * (1 - a2)
    den_s1 = 2 * alpha * (1 - a2)
    
    # Termino s^0: 1 + a1 + a2
    den_s0 = 1 + a1 + a2
    
    # Normalizar para que el coeficiente de mayor orden del denominador sea 1 (si no es 0)
    if abs(den_s2) > 1e-10:
        k = den_s2
    elif abs(den_s1) > 1e-10:
        k = den_s1
    else:
        k = 1.0
    
    num_c , den_c = [num_s2/k, num_s1/k, num_s0/k], [den_s2/k, den_s1/k, den_s0/k]
    den_c[1] = 0.0
    num_c = [num_c[-1]]
        
    return num_c, den_c

def pole_placement(plant, desired_poles):
    """
    Diseño por asignación de polos para sistema en lazo cerrado

    Args:
        desired_poles: Polos deseados del sistema en lazo cerrado

    Returns:
        Kp, Ki, Kd: Parámetros PID
    """
    # Para un controlador PID: C(s) = Kp + Ki/s + Kd*s
    # Sistema en lazo cerrado: T(s) = C(s)G(s)/(1 + C(s)G(s))

    # Polinomio característico deseado
    if len(desired_poles) == 3:
        # Tres polos: s^3 + a*s^2 + b*s + c = 0
        poly = np.poly(desired_poles)
        a, b, c = poly[1], poly[2], poly[3]
    else:
        print("Se necesitan 3 polos para sistema de orden 3")
        return None, None, None

    # Para péndulo invertido, el polinomio característico es:
    # s^3 + Kd*B*s^2 + (Kp*B - A)*s + Ki*B = 0
    # donde A = A_theta, B = B_theta

    # Obtener parámetros de la planta
    if hasattr(plant, 'num') and hasattr(plant, 'den'):
        num = np.atleast_1d(np.squeeze(plant.num))
        den = np.atleast_1d(np.squeeze(plant.den))
        A = -den[2] if len(den) > 2 else 0  # Para inestable: den = [1, 0, -A]
        B = num[0] if len(num) > 0 else 1.0
    else:
        A, B = 15.79, 1.46  # Valores típicos del péndulo

    # Resolver el sistema:
    # Kd*B = a
    # Kp*B - A = b
    # Ki*B = c

    Kd = a / B
    Kp = (b + A) / B
    Ki = c / B

    return Kp, Ki, Kd

def simulate_closed_loop_with_initial_condition(pid, t_total, ref_func, initial_theta=0.1, disturbance_func=None, measurement_noise=0.0):
    """
    Simula el sistema no lineal en lazo cerrado con condición inicial theta != 0
    Incluye AUTOTUNING a los 4 segundos.
    """
    N = int(t_total / Ts)
    t_sim = np.arange(N) * Ts

    # Estado completo del sistema no lineal: [x, xdot, theta, thetadot]
    y_full = np.zeros((4, N)) 
    y_full[0, 0] = 0.0        
    y_full[1, 0] = 0.0        
    y_full[2, 0] = initial_theta  
    y_full[3, 0] = 0.0        

    y = np.zeros(N)  # theta
    u = np.zeros(N)  # control
    ref = np.zeros(N) 
    
    y[0] = initial_theta
    pid.reset()
    
    # Variables para identificación RLS en línea
    na, nb = 2, 2
    n_theta = na + nb 
    theta_hat = np.zeros(n_theta) # [a1, a2, b1, b2]
    P = 1000 * np.eye(n_theta)
    
    # Historiales para plotear
    theta_est_history = np.zeros((N, n_theta))
    pid_history = np.zeros((N, 3)) # [Kp, Ki, Kd]
    
    autotuning_done = False
    AUTOTUNE_TIME = 4.0
    autotune_k = int(AUTOTUNE_TIME / Ts)

    print(f"Iniciando simulación... Autotuning programado para t={AUTOTUNE_TIME}s")

    for k in range(N):
        # Guardar valores actuales de PID
        pid_history[k] = [pid.Kp, pid.Ki, pid.Kd]
        theta_est_history[k] = theta_hat
        
        # Referencia
        ref[k] = ref_func(t_sim[k])

        # Medición con ruido
        if measurement_noise > 0.0:
            y_measured = y[k] + np.random.normal(0, measurement_noise)
        else:
            y_measured = y[k]

        # Error
        error = y_measured - ref[k]

        # Control discreto
        u_control = pid.control(error)
        u[k] = u_control

        
        if k >= max(na, nb) + 1 and k >= autotune_k:
            # Vector de regresores phi = [-y[k-1], -y[k-2], u[k-1], u[k-2]]
            # Notar que u[k-1] es el control aplicado en el paso anterior
            phi = np.array([-y[k-1], -y[k-2], u[k-1], u[k-2]])
            
            # Actualizar estimación
            theta_hist, P, err, ree = estimador_RLS_lazo_cerrado(u[:k], y[:k])
        
        # --- AUTOTUNING TRIGGER ---
        if not autotuning_done and k >= autotune_k:
            #print(f"\n[t={t_sim[k]:.2f}s] Ejecutando AUTOTUNING...")
            
            # Extraer parámetros identificados
            a1_est, a2_est, b1_est, b2_est = theta_hist[-1,0], theta_hist[-1,1], -theta_hist[-1,2], -theta_hist[-1,3]
            theta_hat = [a1_est, a2_est, b1_est, b2_est]
            if k == autotune_k:
                theta_est_history[1:len(theta_hist)+1,:] = theta_hist.copy()
            #print(f"Parámetros estimados: a1={a1_est:.4f}, a2={a2_est:.4f}, b1={b1_est:.4f}, b2={b2_est:.4f}")
            
            # Convertir a continuo
            try:
                num_c_est, den_c_est = conversion_tustin_manual(a1_est, a2_est, b1_est, b2_est, Ts)
                Gs_ident = ctrl.TransferFunction(num_c_est, den_c_est)
                #print(f"Planta identificada G(s): {Gs_ident}")
                
                # Calcular nuevos PID
                new_Kp, new_Ki, new_Kd = pole_placement(Gs_ident, [-3, -3, -10])
                
                if new_Kp is not None:
                    #print(f"Nuevas ganancias PID: Kp={new_Kp:.4f}, Ki={new_Ki:.4f}, Kd={new_Kd:.4f}")
                    # Actualizar controlador
                    pid.Kp = new_Kp
                    pid.Ki = new_Ki
                    pid.Kd = new_Kd

                    autotuning_done = True
                else:
                    print("Error en cálculo de PID. Manteniendo anteriores.")
            except Exception as e:
                print(f"Fallo en autotuning: {e}")
                
        if  autotuning_done and k%10 == 0:#reset autotune
            autotuning_done = False

        # Integración del modelo no lineal
        if k < N-1:
            def ode_with_control(t, y_state):
                return cartpole_ode(t, y_state, u_control)

            t_span = [t_sim[k], t_sim[k+1]]
            sol = solve_ivp(ode_with_control, t_span, y_full[:, k],
                          t_eval=[t_sim[k+1]], method='RK45', rtol=1e-8)

            if sol.success:
                y_full[:, k+1] = sol.y[:, -1]
                if disturbance_func:
                    disturbance = disturbance_func(t_sim[k+1])
                    y_full[2, k+1] += disturbance 
            else:
                y_full[:, k+1] = y_full[:, k]

            y[k+1] = y_full[2, k+1]
    
    return t_sim, y, u, ref, theta_est_history, pid_history

def plot_sim_results(t, y, u, ref, theta_hist, pid_hist, title, subplot_offset):
    # Gráficos de respuesta temporal
    plt.figure()
    subplot_offset=1
    plt.subplot(2, 2, subplot_offset)
    plt.plot(t, y, 'b-', linewidth=1.5, label='theta')
    plt.plot(t, ref, 'k--', linewidth=1, label='Ref')
    plt.axvline(x=4.0, color='m', linestyle=':', label='Autotune')
    plt.ylabel('Theta [rad]')
    plt.title(f'{title} - Respuesta')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(2, 2, subplot_offset + 1)
    plt.plot(t, u, 'r-', linewidth=1.5, label='Control u')
    plt.axvline(x=4.0, color='m', linestyle=':')
    plt.ylabel('u')
    plt.title('Acción de Control')
    plt.grid(True)

    # Evolución de parámetros estimados
    plt.subplot(2, 2, subplot_offset + 2)
    labels = ['$a_1$', '$a_2$', '$b_1$', '$b_2$']
    colors = ['c', 'm', 'y', 'k']
    for i in range(4):
        plt.plot(t, theta_hist[:, i], color=colors[i], label=labels[i], linewidth=1)
    plt.axvline(x=4.0, color='r', linestyle='--')
    plt.ylabel('Valor')
    plt.title('Estimación RLS')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small', ncol=2)

    # Evolución de ganancias PID
    plt.subplot(2, 2, subplot_offset + 3)
    plt.plot(t, pid_hist[:, 0], label='Kp')
    plt.plot(t, pid_hist[:, 1], label='Ki')
    plt.plot(t, pid_hist[:, 2], label='Kd')
    plt.axvline(x=4.0, color='r', linestyle='--')
    plt.ylabel('Ganancia')
    plt.title('Ganancias PID')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'autotuning_completo_{title}.png'), dpi=300)
    print(f"\nResultados guardados en {output_dir}")
    plt.show()

def main():
    output_dir = "presentacion_pid_pendulo/imagenes_pid_autotunning"
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== SIMULACIÓN: AUTOTUNING PID ONLINE ===")

    # PID Inicial (Desintonizado intencionalmente o conservador)
    Kp_ini, Ki_ini, Kd_ini = 40.0, 1.0, 5.0 
    print(f"PID Inicial: Kp={Kp_ini}, Ki={Ki_ini}, Kd={Kd_ini}")
    
    T_SIM = 20.0
    INITIAL_THETA = 0.1
    


    # Guardar datos de simulación
    save_dir = "saved_runs_pid_autotunning"
    os.makedirs(save_dir, exist_ok=True)

    def save_run_data(filename, t, y, u, ref, theta_hist, pid_hist):
        # theta_hist: [a1, a2, b1, b2]
        # pid_hist: [Kp, Ki, Kd]
        df = pd.DataFrame({
            't': t,
            'y': y,
            'u': u,
            'ref': ref,
            'a1_est': theta_hist[:, 0],
            'a2_est': theta_hist[:, 1],
            'b1_est': theta_hist[:, 2],
            'b2_est': theta_hist[:, 3],
            'Kp': pid_hist[:, 0],
            'Ki': pid_hist[:, 1],
            'Kd': pid_hist[:, 2]
        })
        path = os.path.join(save_dir, filename)
        df.to_csv(path, index=False)
        print(f"Datos guardados en {path}")

    # 1. Sin perturbaciones
    pid = PIDController(Kp=Kp_ini, Ki=Ki_ini, Kd=Kd_ini)
    t1, y1, u1, ref1, th1, pid_h1 = simulate_closed_loop_with_initial_condition(
        pid, t_total=T_SIM, ref_func=lambda t: 0.0, initial_theta=INITIAL_THETA
    )
    plot_sim_results(t1, y1, u1, ref1, th1, pid_h1, "Sin Perturbaciones", 1)
    save_run_data('sin_perturbaciones.csv', t1, y1, u1, ref1, th1, pid_h1)

    # 2. Perturbación Sinusoidal
    pid = PIDController(Kp=Kp_ini, Ki=Ki_ini, Kd=Kd_ini)
    t2, y2, u2, ref2, th2, pid_h2 = simulate_closed_loop_with_initial_condition(
        pid, t_total=T_SIM, ref_func=lambda t: 0.0, initial_theta=INITIAL_THETA,
        disturbance_func=lambda t: 0.005 * np.sin(2 * np.pi * 0.5 * t)
    )
    plot_sim_results(t2, y2, u2, ref2, th2, pid_h2, "Pert. Sinusoidal", 5)
    save_run_data('pert_sinusoidal.csv', t2, y2, u2, ref2, th2, pid_h2)
    
    # 3. Perturbación Escalón
    pid = PIDController(Kp=Kp_ini, Ki=Ki_ini, Kd=Kd_ini)
    t3, y3, u3, ref3, th3, pid_h3 = simulate_closed_loop_with_initial_condition(
        pid, t_total=T_SIM, ref_func=lambda t: 0.0, initial_theta=INITIAL_THETA,
        disturbance_func=lambda t: 0.01 if t > 6.0 and t < 8.0 else 0.0
    )
    plot_sim_results(t3, y3, u3, ref3, th3, pid_h3, "Pert. Escalón", 9)
    save_run_data('pert_escalon.csv', t3, y3, u3, ref3, th3, pid_h3)

    # 4. Ruido de Medición
    pid = PIDController(Kp=Kp_ini, Ki=Ki_ini, Kd=Kd_ini)
    t4, y4, u4, ref4, th4, pid_h4 = simulate_closed_loop_with_initial_condition(
        pid, t_total=T_SIM, ref_func=lambda t: 0.0, initial_theta=INITIAL_THETA,
        measurement_noise=0.005
    )
    plot_sim_results(t4, y4, u4, ref4, th4, pid_h4, "Ruido Medición", 13)
    save_run_data('ruido_medicion.csv', t4, y4, u4, ref4, th4, pid_h4)

    

if __name__ == "__main__":
    main()

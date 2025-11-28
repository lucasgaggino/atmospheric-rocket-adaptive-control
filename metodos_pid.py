import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import math

# ============================================
# MÉTODOS PARA CÁLCULO DE PARÁMETROS PID
# ============================================

class PIDTuningMethods:
    """
    Clase que implementa diferentes métodos para calcular parámetros PID
    basados en las características de la planta
    """

    def __init__(self, plant_model, dt=0.02):
        """
        Inicializa con el modelo de la planta

        Args:
            plant_model: Función de transferencia o modelo de la planta
            dt: Tiempo de muestreo para discretización
        """
        self.plant = plant_model
        self.dt = dt

        # Si es función de transferencia continua, discretizar
        if hasattr(plant_model, 'num') and hasattr(plant_model, 'den'):
            self.plant_d = ctrl.c2d(plant_model, dt, method='zoh')
        else:
            self.plant_d = plant_model

    # ============================================
    # MÉTODOS EMPÍRICOS/CLÁSICOS
    # ============================================

    def ziegler_nichols_step(self, Kp_critical=None, Tu=None):
        """
        Método de Ziegler-Nichols basado en respuesta al escalón

        Args:
            Kp_critical: Ganancia crítica (si se conoce)
            Tu: Tiempo de oscilación (si se conoce)

        Returns:
            Kp, Ki, Kd: Parámetros PID
        """
        if Kp_critical is None or Tu is None:
            print("ZN Step requiere Kp_critical y Tu. Use ZN Relay o proporcione valores.")
            return None, None, None

        # Reglas clásicas de ZN
        Kp = 0.6 * Kp_critical
        Ti = 0.5 * Tu
        Td = 0.125 * Tu

        Ki = Kp / Ti
        Kd = Kp * Td

        return Kp, Ki, Kd

    def ziegler_nichols_relay(self, amplitude=0.1, max_iter=50):
        """
        Ziegler-Nichols por relé: experimento de oscilación forzada

        Args:
            amplitude: Amplitud del relé
            max_iter: Máximo número de iteraciones

        Returns:
            Kp, Ki, Kd: Parámetros PID
        """
        # Implementación simplificada del método por relé
        # En la práctica requiere simulación del sistema con relé

        print("ZN Relay: Implementación simplificada")
        # Valores típicos para péndulo invertido (estimados)
        Kp_critical = 50.0  # Ganancia crítica aproximada
        Tu = 0.5  # Periodo de oscilación aproximado

        Kp = 0.6 * Kp_critical
        Ti = 0.5 * Tu
        Td = 0.125 * Tu

        Ki = Kp / Ti
        Kd = Kp * Td

        return Kp, Ki, Kd

    def cohen_coon(self):
        """
        Método de Cohen-Coon para procesos de primer orden con delay
        """
        # Requiere identificación del modelo: K, τ, θ
        # G(s) = K * exp(-θs) / (τs + 1)

        # Para péndulo (aproximado como segundo orden)
        print("Cohen-Coon: Adaptado para sistemas de segundo orden")

        # Parámetros típicos
        Kp = 1.0
        Ti = 1.0
        Td = 0.25

        Ki = Kp / Ti
        Kd = Kp * Td

        return Kp, Ki, Kd

    # ============================================
    # MÉTODOS ANALÍTICOS
    # ============================================

    def pole_placement(self, desired_poles):
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
        if hasattr(self.plant, 'num') and hasattr(self.plant, 'den'):
            num = np.atleast_1d(np.squeeze(self.plant.num))
            den = np.atleast_1d(np.squeeze(self.plant.den))
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

    def internal_model_control(self, tau_c=1.0):
        """
        Método IMC (Internal Model Control)

        Args:
            tau_c: Constante de tiempo del filtro IMC

        Returns:
            Kp, Ki, Kd: Parámetros PID
        """
        # Para procesos de segundo orden: G(s) = B / (s^2 - A)

        # Obtener parámetros
        if hasattr(self.plant, 'num') and hasattr(self.plant, 'den'):
            num = np.atleast_1d(np.squeeze(self.plant.num))
            den = np.atleast_1d(np.squeeze(self.plant.den))
            A = -den[2] if len(den) > 2 else 0
            B = num[0] if len(num) > 0 else 1.0
        else:
            A, B = 15.79, 1.46

        # Para IMC con filtro: C(s) = (s^2 - A)/(B*(τ_c*s + 1))
        # Luego se aproxima a PID

        # Aproximación PID del IMC
        Kp = (2*tau_c*A + 1)/(2*B*tau_c)
        Ki = A / B
        Kd = tau_c / (2*B)

        return Kp, Ki, Kd

    # ============================================
    # MÉTODOS DE OPTIMIZACIÓN
    # ============================================

    def optimize_pid(self, cost_function, initial_guess=[1.0, 0.1, 0.01],
                     bounds=[(0.1, 100), (0.01, 10), (0.001, 1.0)]):
        """
        Optimización numérica de parámetros PID

        Args:
            cost_function: Función de costo (ej: ITAE, ISE)
            initial_guess: Valores iniciales [Kp, Ki, Kd]
            bounds: Límites para los parámetros

        Returns:
            Kp, Ki, Kd: Parámetros óptimos
        """

        def objective(params):
            Kp, Ki, Kd = params
            # Simular sistema con estos parámetros
            cost = cost_function(Kp, Ki, Kd)
            return cost

        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            Kp_opt, Ki_opt, Kd_opt = result.x
            return Kp_opt, Ki_opt, Kd_opt
        else:
            print("Optimización falló")
            return None, None, None

    def itae_optimization(self):
        """
        Optimización usando criterio ITAE (Integral Time Absolute Error)
        """
        def itae_cost(Kp, Ki, Kd):
            # Simular respuesta al escalón
            t = np.linspace(0, 5, 500)
            try:
                # Crear controlador PID
                pid = PIDController(Kp, Ki, Kd, self.dt)

                # Simular respuesta (simplificada)
                y = self.simulate_step_response(pid, t)

                # Calcular ITAE
                error = 1.0 - y  # Referencia = 1
                itae = np.trapz(t * np.abs(error), t)

                return itae
            except:
                return 1e10  # Penalización por inestabilidad

        return self.optimize_pid(itae_cost)

    # ============================================
    # MÉTODOS AUXILIARES
    # ============================================

    def simulate_step_response(self, pid_controller, t):
        """
        Simula respuesta al escalón del sistema controlado
        """
        # Implementación simplificada
        y = np.zeros_like(t)
        error_prev = 0
        integral = 0

        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            error = 1.0 - y[i-1]  # Referencia = 1

            # Control PID
            P = pid_controller.Kp * error
            integral += error * dt
            I = pid_controller.Ki * integral
            D = pid_controller.Kd * (error - error_prev) / dt

            u = P + I + D
            u = np.clip(u, -100, 100)

            # Respuesta del sistema (aproximada)
            # Para péndulo: dy/dt = B*u - A*y (aprox lineal)
            dy = (1.46 * u + 15.79 * y[i-1]) * dt
            y[i] = y[i-1] + dy

            error_prev = error

        return y

    def compare_methods(self, methods=['ZN', 'IMC', 'PolePlacement']):
        """
        Compara diferentes métodos de sintonía PID

        Args:
            methods: Lista de métodos a comparar
        """
        results = {}

        for method in methods:
            if method == 'ZN':
                params = self.ziegler_nichols_relay()
            elif method == 'IMC':
                params = self.internal_model_control()
            elif method == 'PolePlacement':
                # Polos deseados: ζ=0.7, ω_n=5 rad/s
                desired_poles = [-3, -3, -10]  # Tres polos para sistema orden 3
                params = self.pole_placement(desired_poles)
            elif method == 'ITAE':
                params = self.itae_optimization()
            else:
                continue

            if params[0] is not None:
                results[method] = params
                print(f"{method}: Kp={params[0]:.2f}, Ki={params[1]:.2f}, Kd={params[2]:.2f}")

        return results


# ============================================
# CONTROLADOR PID SIMPLE
# ============================================

class PIDController:
    def __init__(self, Kp, Ki, Kd, Ts=0.02):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, error):
        # Término proporcional
        P = self.Kp * error

        # Término integral
        self.integral += error * self.Ts
        I = self.Ki * self.integral

        # Término derivativo
        D = self.Kd * (error - self.prev_error) / self.Ts

        # Control total
        u = P + I + D

        # Saturación
        u = np.clip(u, -100.0, 100.0)

        # Actualizar estado
        self.prev_error = error

        return u


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    # Modelo del péndulo invertido
    M, m, l, g = 1.0, 0.1, 0.5, 9.81
    A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))
    B_theta = 3.0 / (l * (4.0 * M + m))

    # Función de transferencia
    num_c = [B_theta]
    den_c = [1.0, 0.0, -A_theta]
    plant = ctrl.TransferFunction(num_c, den_c)

    # Crear objeto de sintonía
    tuner = PIDTuningMethods(plant, dt=0.02)

    print("=== MÉTODOS DE SINTONÍA PID PARA PÉNDULO INVERTIDO ===")
    print(f"Planta: G(s) = {B_theta:.3f} / (s² - {A_theta:.3f})")
    print()

    # Comparar métodos
    methods = ['ZN', 'IMC', 'PolePlacement']
    results = tuner.compare_methods(methods)

    print()
    print("=== RECOMENDACIONES ===")
    print("• Ziegler-Nichols: Bueno para sistemas oscilatorios")
    print("• IMC: Robusto, buen rechazo de perturbaciones")
    print("• Asignación de Polos: Control preciso de la respuesta")
    print("• Optimización (ITAE): Mínima integral del error absoluto")
    print()
    print("Para péndulo invertido, se recomienda IMC o asignación de polos")
    print("ya que permiten especificar estabilidad y respuesta deseada.")

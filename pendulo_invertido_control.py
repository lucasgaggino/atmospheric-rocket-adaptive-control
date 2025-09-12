"""
Modelo Completo de Péndulo Invertido con Control
===============================================

Implementación a nivel universitario del modelado de un péndulo invertido
con las siguientes características:

- Masa del péndulo: 1 kg
- Longitud del brazo: 1 m (sin masa)
- Masa del carro: 0.1 kg
- Acción de control: Fuerza aplicada al carro
- Punto de trabajo: θ = 0 (equilibrio inestable)

El código incluye:
1. Modelo continuo no lineal
2. Linealización alrededor del punto de equilibrio
3. Modelo discreto usando discretización ZOH
4. Simulaciones y validaciones

Autor: Sistema de Control Adaptativo
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.integrate import solve_ivp
import sympy as sp

class PenduloInvertido:
    """
    Clase que modela un péndulo invertido con control por fuerza aplicada al carro.
    """

    def __init__(self, m_pendulo=1.0, L=1.0, m_carro=0.1, g=9.81):
        """
        Inicialización de parámetros del sistema.

        Parámetros:
        -----------
        m_pendulo : float
            Masa del péndulo [kg]
        L : float
            Longitud del brazo del péndulo [m]
        m_carro : float
            Masa del carro [kg]
        g : float
            Aceleración de la gravedad [m/s²]
        """
        self.m = m_pendulo    # masa del péndulo
        self.M = m_carro      # masa del carro
        self.L = L            # longitud del brazo
        self.g = g            # gravedad

        # Parámetros derivados
        self.total_mass = self.m + self.M

        print("=== Parámetros del Sistema ===")
        print(f"Masa del péndulo: {self.m} kg")
        print(f"Masa del carro: {self.M} kg")
        print(f"Longitud del brazo: {self.L} m")
        print(f"Masa total: {self.total_mass} kg")
        print(f"Gravedad: {self.g} m/s²")
        print()

    def modelo_no_lineal(self, t, estado, F):
        """
        Modelo dinámico no lineal del péndulo invertido.

        Estado: [x, x_dot, theta, theta_dot]
        Donde:
        - x: posición del carro [m]
        - x_dot: velocidad del carro [m/s]
        - theta: ángulo del péndulo [rad]
        - theta_dot: velocidad angular [rad/s]

        Ecuaciones:
        -----------
        Para el carro: F = (m+M)*ẍ + m*L*cos(θ)*θ̈ - m*L*sin(θ)*θ̇²
        Para el péndulo: τ = m*g*L*sin(θ) + m*L*cos(θ)*ẍ = m*L²*θ̈

        Parámetros:
        -----------
        t : float
            Tiempo
        estado : array
            Vector de estado [x, x_dot, theta, theta_dot]
        F : float
            Fuerza aplicada al carro

        Retorna:
        --------
        d_estado : array
            Derivada del vector de estado
        """
        x, x_dot, theta, theta_dot = estado

        # Términos trigonométricos
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Aceleración angular del péndulo
        # θ̈ = [g*sin(θ) + cos(θ)*ẍ]/L
        # Pero necesitamos resolver el sistema acoplado

        # Denominador común para resolver θ̈
        denom = self.m * self.L * cos_theta**2 - self.total_mass * self.L

        # Término independiente
        term1 = self.m * self.L * sin_theta * cos_theta * theta_dot**2
        term2 = self.m * self.g * sin_theta * cos_theta
        term3 = F * cos_theta

        # Aceleración angular
        theta_ddot = (term1 + term2 - term3) / denom

        # Aceleración del carro
        # ẍ = [F + m*L*sin(θ)*θ̇² - m*L*cos(θ)*θ̈]/(m+M)
        x_ddot = (F + self.m * self.L * sin_theta * theta_dot**2 -
                 self.m * self.L * cos_theta * theta_ddot) / self.total_mass

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def linealizar_sistema(self):
        """
        Linealización del sistema alrededor del punto de equilibrio.
        Punto de equilibrio: θ = 0, θ̇ = 0, x = 0, ẋ = 0, F = 0

        Las ecuaciones linealizadas son:

        ẍ = [F + m*L*θ̈]/(m+M)    (para θ ≈ 0, sinθ ≈ θ, cosθ ≈ 1)
        θ̈ = [g*θ + ẍ]/L

        Sustituyendo:
        θ̈ = [g*θ + (F + m*L*θ̈)/(m+M)]/L
        θ̈ - (m*L/(m+M))/L * θ̈ = (g/L)*θ + F/((m+M)*L)
        θ̈ = [g*(m+M)*θ + F]/[(m+M)*L - m*L]

        Matriz A = [[0, 1, 0, 0],
                   [0, 0, num1, 0],
                   [0, 0, 0, 1],
                   [0, 0, num2, 0]]

        Matriz B = [[0],
                   [denom1],
                   [0],
                   [denom2]]

        Donde:
        num1 = -m*g/(m+M)
        num2 = g*(m+M)/((m+M)*L - m*L)
        denom1 = 1/(m+M)
        denom2 = 1/((m+M)*L - m*L)
        """
        # Coeficientes para las matrices
        num1 = -self.m * self.g / self.total_mass
        num2 = self.g * self.total_mass / (self.total_mass * self.L - self.m * self.L)
        denom1 = 1 / self.total_mass
        denom2 = 1 / (self.total_mass * self.L - self.m * self.L)

        # Matriz A (sistema linealizado)
        A = np.array([[0, 1, 0, 0],
                      [0, 0, num1, 0],
                      [0, 0, 0, 1],
                      [0, 0, num2, 0]])

        # Matriz B (entrada: fuerza F)
        B = np.array([[0],
                      [denom1],
                      [0],
                      [denom2]])

        # Matriz C (salidas: posición del carro y ángulo del péndulo)
        C = np.array([[1, 0, 0, 0],  # medir posición del carro
                      [0, 0, 1, 0]]) # medir ángulo del péndulo

        # Matriz D
        D = np.array([[0],
                      [0]])

        print("=== Modelo Linealizado ===")
        print("Matriz A:")
        print(A)
        print("\nMatriz B:")
        print(B)
        print("\nMatriz C:")
        print(C)
        print("\nMatriz D:")
        print(D)

        # Verificar controlabilidad
        ctrb_matrix = ctrl.ctrb(A, B)
        rank_ctrb = np.linalg.matrix_rank(ctrb_matrix)
        print(f"\nRango de la matriz de controlabilidad: {rank_ctrb}")
        print(f"Dimensión del sistema: {A.shape[0]}")
        if rank_ctrb == A.shape[0]:
            print("✓ Sistema controlable")
        else:
            print("✗ Sistema no controlable")

        # Verificar observabilidad (para ambas salidas)
        obsv_matrix = ctrl.obsv(A, C)
        rank_obsv = np.linalg.matrix_rank(obsv_matrix)
        print(f"\nRango de la matriz de observabilidad: {rank_obsv}")
        if rank_obsv == A.shape[0]:
            print("✓ Sistema observable")
        else:
            print("✗ Sistema no observable")

        return A, B, C, D

    def crear_sistema_continuo(self):
        """
        Crea el sistema continuo usando python-control.
        """
        A, B, C, D = self.linealizar_sistema()

        # Sistema en espacio de estados
        sys_continuo = ctrl.ss(A, B, C, D)

        print("\n=== Sistema Continuo ===")
        print("Función de transferencia:")
        tf_sys = ctrl.ss2tf(sys_continuo)
        print(tf_sys)

        return sys_continuo

    def discretizar_ZOH(self, Ts=0.01):
        """
        Discretiza el sistema continuo usando Zero-Order Hold (ZOH).

        Parámetros:
        -----------
        Ts : float
            Tiempo de muestreo [s]

        Retorna:
        --------
        sys_discreto : control.StateSpace
            Sistema discretizado
        """
        sys_continuo = self.crear_sistema_continuo()

        print(f"\n=== Discretización ZOH (Ts = {Ts} s) ===")

        # Discretización usando ZOH
        sys_discreto = ctrl.c2d(sys_continuo, Ts, method='zoh')

        print("Sistema discretizado:")
        print(f"Matriz A_d:")
        print(sys_discreto.A)
        print(f"\nMatriz B_d:")
        print(sys_discreto.B)
        print(f"\nMatriz C_d:")
        print(sys_discreto.C)
        print(f"\nMatriz D_d:")
        print(sys_discreto.D)

        return sys_discreto

    def simular_respuesta(self, Ts=0.01, T_sim=5.0, F_input=None):
        """
        Simula la respuesta del sistema a una entrada de fuerza.

        Parámetros:
        -----------
        Ts : float
            Tiempo de muestreo
        T_sim : float
            Tiempo total de simulación
        F_input : callable or None
            Función de entrada F(t). Si None, usa un escalón unitario.
        """
        # Crear sistemas
        sys_continuo = self.crear_sistema_continuo()
        sys_discreto = self.discretizar_ZOH(Ts)

        # Definir entrada por defecto (escalón unitario)
        if F_input is None:
            def F_input(t):
                return 1.0 if t >= 0 else 0.0

        # Simulación continua
        t_cont = np.linspace(0, T_sim, 1000)

        # Simulación usando python-control
        u_cont = F_input(t_cont)
        t_cont, y_cont = ctrl.forced_response(sys_continuo, t_cont, U=u_cont)

        # Simulación discreta
        t_disc = np.arange(0, T_sim, Ts)
        u_disc = np.array([F_input(t) for t in t_disc])

        t_disc, y_disc = ctrl.forced_response(sys_discreto, t_disc, U=u_disc)

        # Graficar resultados
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Posición del carro
        axes[0, 0].plot(t_cont, y_cont[0, :], 'b-', label='Continuo', linewidth=2)
        axes[0, 0].step(t_disc, y_disc[0, :], 'r--', label='Discreto (ZOH)', where='post')
        axes[0, 0].set_ylabel('Posición del carro [m]')
        axes[0, 0].set_title('Respuesta de la posición del carro')
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        # Ángulo del péndulo
        axes[0, 1].plot(t_cont, y_cont[1, :], 'b-', label='Continuo', linewidth=2)
        axes[0, 1].step(t_disc, y_disc[1, :], 'r--', label='Discreto (ZOH)', where='post')
        axes[0, 1].set_ylabel('Ángulo θ [rad]')
        axes[0, 1].set_title('Respuesta del ángulo del péndulo')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # Entrada de fuerza
        axes[1, 0].plot(t_cont, u_cont, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Tiempo [s]')
        axes[1, 0].set_ylabel('Fuerza F [N]')
        axes[1, 0].set_title('Entrada de fuerza aplicada')
        axes[1, 0].grid(True)

        # Error de discretización (interpolación simple)
        from scipy import interpolate
        f_pos = interpolate.interp1d(t_cont, y_cont[0, :], kind='linear')
        f_theta = interpolate.interp1d(t_cont, y_cont[1, :], kind='linear')

        y_cont_interp_pos = f_pos(t_disc)
        y_cont_interp_theta = f_theta(t_disc)

        error_pos = y_cont_interp_pos - y_disc[0, :]
        error_theta = y_cont_interp_theta - y_disc[1, :]

        axes[1, 1].plot(t_disc, error_pos, 'b-', label='Error posición')
        axes[1, 1].plot(t_disc, error_theta, 'r-', label='Error ángulo')
        axes[1, 1].set_xlabel('Tiempo [s]')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Error de discretización')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        return t_cont, y_cont, t_disc, y_disc

    def analizar_estabilidad(self):
        """
        Analiza la estabilidad del sistema continuo y discreto.
        """
        sys_continuo = self.crear_sistema_continuo()

        print("\n=== Análisis de Estabilidad ===")

        # Polos del sistema continuo
        polos_cont = sys_continuo.poles()
        print("Polos del sistema continuo:")
        for i, polo in enumerate(polos_cont):
            print(".6f")

        # Verificar estabilidad continua
        stable_cont = all(np.real(polo) < 0 for polo in polos_cont)
        print(f"Estabilidad continua: {'Estable' if stable_cont else 'Inestable'}")

        # Análisis para diferentes tiempos de muestreo
        Ts_values = [0.001, 0.01, 0.1]
        for Ts in Ts_values:
            sys_disc = ctrl.c2d(sys_continuo, Ts, method='zoh')
            polos_disc = sys_disc.poles()

            print(f"\nTs = {Ts} s:")
            print("Polos discretos:")
            for i, polo in enumerate(polos_disc):
                mag = np.abs(polo)
                print(".4f")

            # Verificar estabilidad discreta (|z| < 1)
            stable_disc = all(np.abs(polo) < 1 for polo in polos_disc)
            print(f"Estabilidad discreta: {'Estable' if stable_disc else 'Inestable'}")

    def disenhar_controlador(self, metodo='lqr', Ts=0.01):
        """
        Diseña un controlador para estabilizar el péndulo.

        Parámetros:
        -----------
        metodo : str
            Método de control ('lqr', 'pole_placement')
        Ts : float
            Tiempo de muestreo para el controlador discreto
        """
        sys_continuo = self.crear_sistema_continuo()
        sys_discreto = ctrl.c2d(sys_continuo, Ts, method='zoh')

        if metodo == 'lqr':
            print("\n=== Diseño LQR ===")

            # Matrices de ponderación
            Q = np.diag([100, 1, 1000, 10])  # [x, x_dot, theta, theta_dot]
            R = np.array([[0.1]])  # Control effort

            print("Matriz Q:")
            print(Q)
            print("Matriz R:")
            print(R)

            # Diseño del controlador continuo
            K_cont, S_cont, E_cont = ctrl.lqr(sys_continuo, Q, R)
            print("\nGanancia LQR continua:")
            print(K_cont)
            print("Valores propios de lazo cerrado:")
            print(E_cont)

            # Diseño del controlador discreto
            K_disc, S_disc, E_disc = ctrl.dlqr(sys_discreto, Q, R)
            print("\nGanancia LQR discreta:")
            print(K_disc)
            print("Valores propios de lazo cerrado:")
            print(E_disc)

            return K_cont, K_disc

        elif metodo == 'pole_placement':
            print("\n=== Colocación de Polos ===")

            # Polos deseados (todos con parte real -2)
            polos_deseados = [-2, -2.1, -2.2, -2.3]

            # Colocación de polos continua
            K_cont = ctrl.place(sys_continuo.A, sys_continuo.B, polos_deseados)
            print("Ganancia colocación de polos continua:")
            print(K_cont)

            # Colocación de polos discreta
            polos_deseados_disc = np.exp(np.array(polos_deseados) * Ts)
            K_disc = ctrl.place(sys_discreto.A, sys_discreto.B, polos_deseados_disc)
            print("Ganancia colocación de polos discreta:")
            print(K_disc)

            return K_cont, K_disc


def main():
    """
    Función principal que demuestra el uso de la clase PenduloInvertido.
    """
    print("=== MODELADO DE PÉNDULO INVERTIDO ===")
    print("Implementación a nivel universitario")
    print("=" * 50)

    # Crear instancia del péndulo
    pendulo = PenduloInvertido(m_pendulo=1.0, L=1.0, m_carro=0.1, g=9.81)

    # Análisis de estabilidad
    pendulo.analizar_estabilidad()

    # Simulación de respuesta
    print("\n=== SIMULACIÓN DE RESPUESTA ===")
    Ts = 0.01  # tiempo de muestreo
    T_sim = 3.0  # tiempo de simulación

    # Función de entrada (pulso de fuerza)
    def fuerza_pulso(t):
        if np.isscalar(t):
            if 0.5 <= t <= 1.0:
                return 2.0  # 2 Newtons durante 0.5 segundos
            else:
                return 0.0
        else:
            # Para arrays
            result = np.zeros_like(t)
            mask = (t >= 0.5) & (t <= 1.0)
            result[mask] = 2.0
            return result

    t_cont, y_cont, t_disc, y_disc = pendulo.simular_respuesta(
        Ts=Ts, T_sim=T_sim, F_input=fuerza_pulso
    )

    # Diseño de controlador
    print("\n=== DISEÑO DE CONTROLADOR ===")
    K_lqr_cont, K_lqr_disc = pendulo.disenhar_controlador(metodo='lqr', Ts=Ts)

    print("\n=== RESUMEN ===")
    print("✓ Modelo continuo implementado")
    print("✓ Linealización realizada alrededor de θ = 0")
    print("✓ Discretización ZOH implementada")
    print("✓ Simulaciones realizadas")
    print("✓ Controlador LQR diseñado")
    print("\nEl sistema está listo para análisis y control avanzado.")


if __name__ == "__main__":
    main()

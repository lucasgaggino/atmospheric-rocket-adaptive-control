import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp

# Physical parameters
M = 1.0  # cart mass [kg]
m = 0.1  # pendulum mass [kg]
l = 0.5  # pendulum length [m]
g = 9.81  # gravity [m/s^2]

# Linearized system parameters
A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))
B_theta = -3.0 / (l * (4.0 * M + m))

print("="*60)
print("ANÁLISIS DE INESTABILIDAD DEL PÉNDULO INVERTIDO")
print("="*60)

# Test 1: Discretization with DIFFERENT sampling times
print("\n1. DISCRETIZACIÓN CON DIFERENTES Ts:")
print("-" * 60)

sampling_times = [0.001, 0.02, 0.1, 0.5]

for Ts in sampling_times:
    num_c = [B_theta]
    den_c = [1.0, 0.0, -A_theta]
    sys_d = cont2discrete((num_c, den_c), Ts, method='zoh')
    A_d = sys_d[0][0][0]
    B_d = sys_d[0][0][1]
    
    print(f"Ts = {Ts:5.3f} s  →  A_d = {A_d:10.6f}, B_d = {B_d:10.6f}")
    
    if abs(A_d) > 1:
        print(f"              ⚠️  INESTABLE (|A_d| = {abs(A_d):.3f} > 1)")
    else:
        print(f"              ✓  Aparentemente estable (|A_d| = {abs(A_d):.3f} < 1)")

# Test 2: Simulate with small perturbation (continuous system)
print("\n2. SIMULACIÓN CONTINUA CON PEQUEÑA PERTURBACIÓN:")
print("-" * 60)

def linearized_pendulum(t, y):
    """Linearized continuous dynamics"""
    theta, theta_dot = y
    # No control input (u=0), solo condición inicial
    theta_ddot = A_theta * theta  # Unstable!
    return [theta_dot, theta_ddot]

# Initial condition: small angle deviation
theta_0 = 0.01  # 0.01 rad ≈ 0.57°
t_span = (0, 3.0)
t_eval = np.linspace(0, 3.0, 1000)

sol = solve_ivp(linearized_pendulum, t_span, [theta_0, 0], 
                t_eval=t_eval, method='RK45')

print(f"Condición inicial: θ(0) = {theta_0} rad ({theta_0*180/np.pi:.2f}°)")
print(f"Después de 1.0 s: θ = {sol.y[0, np.argmin(np.abs(sol.t-1.0))]:.4f} rad")
print(f"Después de 2.0 s: θ = {sol.y[0, np.argmin(np.abs(sol.t-2.0))]:.4f} rad")
print(f"Después de 3.0 s: θ = {sol.y[0, np.argmin(np.abs(sol.t-3.0))]:.4f} rad")
print("⚠️  ¡El ángulo DIVERGE exponencialmente! (Sistema INESTABLE)")

# Test 3: Discrete simulation with Ts=0.02 vs Ts=0.1
print("\n3. COMPARACIÓN DISCRETA (Ts=0.02 vs Ts=0.1):")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Continuous system
axes[0, 0].plot(sol.t, sol.y[0], 'r-', linewidth=2)
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('Angle [rad]')
axes[0, 0].set_title('Sistema Continuo (INESTABLE)\nθ(0)=0.01 rad, u=0')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)

# Discrete with Ts=0.02 (your case)
Ts1 = 0.02
sys_d1 = cont2discrete((num_c, den_c), Ts1, method='zoh')
A_d1, B_d1 = sys_d1[0][0][0], sys_d1[0][0][1]

N1 = 150
theta_disc1 = np.zeros(N1)
theta_disc1[0] = theta_0
for k in range(1, N1):
    theta_disc1[k] = A_d1 * theta_disc1[k-1]  # u=0

t_disc1 = np.arange(N1) * Ts1

axes[0, 1].plot(t_disc1, theta_disc1, 'b.-', linewidth=1.5, markersize=3)
axes[0, 1].set_xlabel('Time [s]')
axes[0, 1].set_ylabel('Angle [rad]')
axes[0, 1].set_title(f'Discreto Ts={Ts1}s (A_d={A_d1:.6f})\nParece ESTABLE pero es artefacto')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)

# Discrete with Ts=0.1 (larger)
Ts2 = 0.1
sys_d2 = cont2discrete((num_c, den_c), Ts2, method='zoh')
A_d2, B_d2 = sys_d2[0][0][0], sys_d2[0][0][1]

N2 = 30
theta_disc2 = np.zeros(N2)
theta_disc2[0] = theta_0
for k in range(1, N2):
    theta_disc2[k] = A_d2 * theta_disc2[k-1]  # u=0

t_disc2 = np.arange(N2) * Ts2

axes[1, 0].plot(t_disc2, theta_disc2, 'g.-', linewidth=1.5, markersize=5)
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_ylabel('Angle [rad]')
axes[1, 0].set_title(f'Discreto Ts={Ts2}s (A_d={A_d2:.6f})\nClaramente INESTABLE (|A_d|>1)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)

# Comparison
axes[1, 1].plot(sol.t, sol.y[0], 'r-', linewidth=2, label='Continuo (real)')
axes[1, 1].plot(t_disc1, theta_disc1, 'b.-', linewidth=1, markersize=2, label=f'Discreto Ts={Ts1}s')
axes[1, 1].plot(t_disc2, theta_disc2, 'g.-', linewidth=1, markersize=4, label=f'Discreto Ts={Ts2}s')
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_ylabel('Angle [rad]')
axes[1, 1].set_title('Comparación de Respuestas')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_inestabilidad.png', dpi=150, bbox_inches='tight')
print("\n✓ Gráficos guardados en 'analisis_inestabilidad.png'")
plt.show()

print("\n" + "="*60)
print("CONCLUSIÓN:")
print("="*60)
print("El péndulo invertido ES INESTABLE en su forma continua.")
print("Con Ts=0.02s, la discretización OCULTA la inestabilidad (A_d≈0).")
print("Esto NO representa la realidad física del sistema.")
print("Para identificación realista, se necesita:")
print("  1. Ts más grande (0.1s) para capturar inestabilidad")
print("  2. Simulación en lazo cerrado con controlador")
print("  3. Datos experimentales reales")
print("="*60)


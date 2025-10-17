import numpy as np
import matplotlib.pyplot as plt

# Physical parameters (same as pendulo.py)
M = 1.0  # cart mass [kg]
m = 0.1  # pendulum mass [kg]
l = 0.5  # pendulum length [m]
g = 9.81  # gravity [m/s^2]

# Linearized system parameters (from pendulo.py)
A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))  # unstable pole
B_theta = -3.0 / (l * (4.0 * M + m))

# Discretization (ZOH, same as pendulo.py)
Ts = 0.02  # sampling time [s]
from scipy.signal import cont2discrete
num_c = [B_theta]
den_c = [1.0, 0.0, -A_theta]
sys_d = cont2discrete((num_c, den_c), Ts, method='zoh')
A_d = sys_d[0][0][0]  # extract discrete A (should be close to 1)
B_d = sys_d[0][0][1]  # extract discrete B

print(f"Continuous: poles at ±{np.sqrt(A_theta):.3f}j")
print(f"Discrete: A={A_d:.6f}, B={B_d:.6f}")

# Generate identification data using discrete model
np.random.seed(42)
N = 300

# PRBS input with longer switching times (more readable)
u = np.zeros(N)
switch_time = 15  # samples per switch
for i in range(0, N, switch_time):
    u[i:i+switch_time] = np.random.choice([-5, 5])

# Simulate discrete system
theta = np.zeros(N)
for k in range(2, N):
    theta[k] = A_d * theta[k-1] + B_d * u[k-1]

# ========================================
# LINEAR ARX MODEL IDENTIFICATION
# ========================================
# Model: theta[k] = a1*theta[k-1] + b1*u[k-1]
print("\n" + "="*50)
print("LINEAR ARX MODEL IDENTIFICATION")
print("="*50)

k_start = 1
X_linear = []
y_target = []

for k in range(k_start, N):
    X_linear.append([theta[k-1], u[k-1]])
    y_target.append(theta[k])

X_linear = np.array(X_linear)
y_target = np.array(y_target)

# Least squares for linear model
theta_hat_linear = np.linalg.pinv(X_linear) @ y_target

print("Estimated parameters (Linear ARX):")
print(f"  a1 (theta[k-1]):    {theta_hat_linear[0]:.6f}")
print(f"  b1 (u[k-1]):        {theta_hat_linear[1]:.6f}")
print(f"True parameters: a1={A_d:.6f}, b1={B_d:.6f}")

# Validation: simulate with linear model
theta_sim_linear = np.zeros(N)
theta_sim_linear[0] = theta[0]

for k in range(k_start, N):
    theta_sim_linear[k] = theta_hat_linear[0]*theta_sim_linear[k-1] + theta_hat_linear[1]*u[k-1]

# RMSE for linear model
rmse_linear = np.sqrt(np.mean((theta - theta_sim_linear)**2))
print(f"RMSE (Linear): {rmse_linear:.6f}")

# ========================================
# NONLINEAR NARX MODEL IDENTIFICATION
# ========================================
# Model: theta[k] = a1*theta[k-1] + a2*sin(theta[k-1]) + b1*u[k-1] + b2*u[k-1]^2
print("\n" + "="*50)
print("NONLINEAR NARX MODEL IDENTIFICATION")
print("="*50)

X_nonlinear = []
for k in range(k_start, N):
    X_nonlinear.append([
        theta[k-1],              # linear term
        np.sin(theta[k-1]),      # nonlinear term: sin(theta)
        u[k-1],                   # linear input
        u[k-1]**2                 # nonlinear input: u^2
    ])

X_nonlinear = np.array(X_nonlinear)

# Least squares for nonlinear model
theta_hat_nonlinear = np.linalg.pinv(X_nonlinear) @ y_target

print("Estimated parameters (Nonlinear NARX):")
print(f"  a1 (theta[k-1]):     {theta_hat_nonlinear[0]:.6f}")
print(f"  a2 (sin(theta[k-1])): {theta_hat_nonlinear[1]:.6f}")
print(f"  b1 (u[k-1]):         {theta_hat_nonlinear[2]:.6f}")
print(f"  b2 (u[k-1]^2):       {theta_hat_nonlinear[3]:.6f}")

# Validation: simulate with nonlinear model
theta_sim_nonlinear = np.zeros(N)
theta_sim_nonlinear[0] = theta[0]

for k in range(k_start, N):
    theta_sim_nonlinear[k] = (theta_hat_nonlinear[0]*theta_sim_nonlinear[k-1] + 
                               theta_hat_nonlinear[1]*np.sin(theta_sim_nonlinear[k-1]) + 
                               theta_hat_nonlinear[2]*u[k-1] + 
                               theta_hat_nonlinear[3]*u[k-1]**2)

# RMSE for nonlinear model
rmse_nonlinear = np.sqrt(np.mean((theta - theta_sim_nonlinear)**2))
print(f"RMSE (Nonlinear): {rmse_nonlinear:.6f}")

print("\n" + "="*50)
print(f"RMSE Improvement: {(rmse_linear - rmse_nonlinear)/rmse_linear * 100:.2f}%")
print("="*50)

# ========================================
# PLOTTING RESULTS
# ========================================
t = np.arange(N) * Ts

# Figure 1: Input and Output Comparison
plt.figure(figsize=(15, 9))

plt.subplot(3, 2, 1)
plt.step(t, u, where='post')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.title('Input Signal (PRBS)')
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 2)
plt.plot(t, theta, 'k-', label='True', linewidth=2)
plt.plot(t, theta_sim_linear, 'b--', label='Linear ARX', linewidth=1.5, alpha=0.7)
plt.plot(t, theta_sim_nonlinear, 'r:', label='Nonlinear NARX', linewidth=1.5, alpha=0.7)
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.title('Output Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 3)
plt.plot(t, theta - theta_sim_linear, 'b-', label='Linear ARX Error')
plt.xlabel('Time [s]')
plt.ylabel('Error [rad]')
plt.title(f'Linear ARX Prediction Error (RMSE={rmse_linear:.6f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 4)
plt.plot(t, theta - theta_sim_nonlinear, 'r-', label='Nonlinear NARX Error')
plt.xlabel('Time [s]')
plt.ylabel('Error [rad]')
plt.title(f'Nonlinear NARX Prediction Error (RMSE={rmse_nonlinear:.6f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 5)
# Zoom in on a section to see differences
zoom_start, zoom_end = 50, 150
plt.plot(t[zoom_start:zoom_end], theta[zoom_start:zoom_end], 'k-', label='True', linewidth=2)
plt.plot(t[zoom_start:zoom_end], theta_sim_linear[zoom_start:zoom_end], 'b--', 
         label='Linear ARX', linewidth=1.5)
plt.plot(t[zoom_start:zoom_end], theta_sim_nonlinear[zoom_start:zoom_end], 'r:', 
         label='Nonlinear NARX', linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.title('Output Comparison (Zoomed)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 6)
# Error comparison bar chart
models = ['Linear\nARX', 'Nonlinear\nNARX']
rmse_values = [rmse_linear, rmse_nonlinear]
colors = ['blue', 'red']
bars = plt.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('RMSE [rad]')
plt.title('RMSE Comparison')
plt.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, rmse in zip(bars, rmse_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{rmse:.6f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('identification_comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'identification_comparison.png'")
plt.show()

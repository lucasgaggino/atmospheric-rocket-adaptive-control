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

# ARX model identification (simplified)
# Model: theta[k] = a1*theta[k-1] + b1*u[k-1]
k_start = 1
X = []
y_target = []

for k in range(k_start, N):
    X.append([theta[k-1], u[k-1]])
    y_target.append(theta[k])

X = np.array(X)
y_target = np.array(y_target)

# Least squares
theta_hat = np.linalg.pinv(X) @ y_target

print("Estimated parameters:")
print(f"a1 (theta[k-1]): {theta_hat[0]:.4f}")
print(f"b1 (u[k-1]): {theta_hat[1]:.4f}")
print(f"True parameters: a1={A_d:.4f}, b1={B_d:.4f}")

# Validation: simulate with estimated model
theta_sim = np.zeros(N)
theta_sim[0] = theta[0]

for k in range(k_start, N):
    theta_sim[k] = theta_hat[0]*theta_sim[k-1] + theta_hat[1]*u[k-1]

# Plot results
t = np.arange(N) * Ts

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.step(t, u)
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.title('Input')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(t, theta, label='True')
plt.plot(t, theta_sim, '--', label='Estimated')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.title('Output')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(t, theta - theta_sim)
plt.xlabel('Time [s]')
plt.ylabel('Error [rad]')
plt.title('Prediction Error')
plt.grid(True)

plt.tight_layout()
plt.show()

# RMSE
rmse = np.sqrt(np.mean((theta - theta_sim)**2))
print(f"RMSE: {rmse:.6f}")

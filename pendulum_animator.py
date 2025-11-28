import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import control as ctrl
import os

class PendulumAnimator:
    """
    Clase para crear animaciones del péndulo invertido con control y perturbaciones
    """

    def __init__(self, pendulum_model, controller, t_total=10.0, dt=0.02,
                 initial_theta=0.1, reference_func=lambda t: 0.0,
                 disturbance_func=None, cart_width=0.3, pendulum_length=0.5):
        """
        Inicializa el animador del péndulo

        Args:
            pendulum_model: Modelo del péndulo (TransferFunction o parámetros)
            controller: Controlador (función o clase con método control)
            t_total: Tiempo total de simulación
            dt: Paso de tiempo
            initial_theta: Ángulo inicial del péndulo
            reference_func: Función de referencia (theta_ref(t))
            disturbance_func: Función de perturbación (disturbance(t))
            cart_width: Ancho del carro
            pendulum_length: Longitud del péndulo
        """
        self.pendulum_model = pendulum_model
        self.controller = controller
        self.t_total = t_total
        self.dt = dt
        self.initial_theta = initial_theta
        self.reference_func = reference_func
        self.disturbance_func = disturbance_func
        self.cart_width = cart_width
        self.pendulum_length = pendulum_length

        # Parámetros del péndulo
        self.M = 1.0  # masa del carro [kg]
        self.m = 0.1  # masa del péndulo [kg]
        self.l = pendulum_length  # longitud del péndulo [m]
        self.g = 9.81  # gravedad [m/s^2]

        # Constantes del modelo linealizado
        self.A_theta = 3.0 * self.g * (self.M + self.m) / (self.l * (4.0 * self.M + self.m))
        self.B_theta = 3.0 / (self.l * (4.0 * self.M + self.m))

        # Simular el sistema
        self.simulate_system()

        # Configurar la animación
        self.setup_animation()

    def simulate_system(self):
        """Simula el sistema con control y perturbaciones"""
        # Crear modelo discreto
        num_c = [self.B_theta]
        den_c = [1.0, 0.0, -self.A_theta]
        G_s = ctrl.TransferFunction(num_c, den_c)
        G_z = ctrl.c2d(G_s, self.dt, method="zoh")
        G_z_ss = ctrl.tf2ss(G_z)

        # Inicialización
        N = int(self.t_total / self.dt)
        self.t = np.arange(N) * self.dt
        self.theta = np.zeros(N)
        self.theta_dot = np.zeros(N)  # velocidad angular (estimada)
        self.u = np.zeros(N)
        self.ref = np.zeros(N)
        self.disturbance = np.zeros(N)

        # Estado del sistema
        x = np.zeros((G_z_ss.A.shape[0], 1))
        self.theta[0] = self.initial_theta

        # Reset del controlador si tiene método reset
        if hasattr(self.controller, 'reset'):
            self.controller.reset()

        for k in range(N):
            # Referencia
            self.ref[k] = self.reference_func(self.t[k])

            # Error
            error = self.ref[k] - self.theta[k-1] if k > 0 else self.ref[k] - self.initial_theta

            # Control
            if callable(self.controller):
                # Si es una función
                self.u[k] = self.controller(error)
            elif hasattr(self.controller, 'control'):
                # Si es una clase con método control
                self.u[k] = self.controller.control(error)
            else:
                self.u[k] = 0.0

            # Perturbación
            self.disturbance[k] = self.disturbance_func(self.t[k]) if self.disturbance_func else 0.0

            # Sistema (solo para k > 0)
            if k > 0:
                x = G_z_ss.A @ x + G_z_ss.B * self.u[k]
                self.theta[k] = float(G_z_ss.C @ x) + self.disturbance[k]

                # Estimar velocidad angular (diferencia finita)
                self.theta_dot[k] = (self.theta[k] - self.theta[k-1]) / self.dt

        # Calcular posiciones cartesianas
        self.cart_x = np.zeros_like(self.theta)  # carro fijo en x=0
        self.pendulum_x = self.cart_x + self.l * np.sin(self.theta)
        self.pendulum_y = -self.l * np.cos(self.theta)  # negativo para que baje

    def setup_animation(self):
        """Configura la figura y elementos de la animación"""
        # Crear figura con subplots
        self.fig = plt.figure(figsize=(15, 8))

        # Subplot 1: Animación del péndulo
        self.ax1 = plt.subplot(2, 3, (1, 4))
        self.ax1.set_xlim(-1.5, 1.5)
        self.ax1.set_ylim(-1.2, 0.8)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Animación del Péndulo Invertido')
        self.ax1.grid(True, alpha=0.3)

        # Elementos del péndulo
        self.cart, = self.ax1.plot([], [], 's-', linewidth=4, markersize=20, color='blue', label='Carro')
        self.pendulum_line, = self.ax1.plot([], [], 'o-', linewidth=3, markersize=8, color='red', label='Péndulo')
        self.pivot, = self.ax1.plot([0], [0], 'ko', markersize=6, label='Pivote')

        # Fuerzas/acciones
        self.force_arrow = self.ax1.arrow(0, 0.3, 0, 0, head_width=0.05, head_length=0.05,
                                        fc='green', ec='green', alpha=0.7, label='Fuerza de control')
        self.disturbance_arrow = self.ax1.arrow(0, -0.3, 0, 0, head_width=0.05, head_length=0.05,
                                              fc='orange', ec='orange', alpha=0.7, label='Perturbación')

        self.ax1.legend(loc='upper right')

        # Subplot 2: Ángulo vs tiempo
        self.ax2 = plt.subplot(2, 3, 5)
        self.theta_line, = self.ax2.plot([], [], 'b-', linewidth=2, label='θ(t)')
        self.ref_line, = self.ax2.plot([], [], 'k--', linewidth=1, label='Referencia')
        self.ax2.set_xlim(0, self.t_total)
        self.ax2.set_ylim(-0.5, 0.5)
        self.ax2.set_xlabel('Tiempo [s]')
        self.ax2.set_ylabel('Ángulo θ [rad]')
        self.ax2.set_title('Evolución del Ángulo')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()

        # Subplot 3: Acción de control
        self.ax3 = plt.subplot(2, 3, 6)
        self.control_line, = self.ax3.plot([], [], 'r-', linewidth=2, label='u(t)')
        self.disturbance_line, = self.ax3.plot([], [], color='orange', linewidth=1, label='Perturbación', alpha=0.7)
        self.ax3.set_xlim(0, self.t_total)
        control_max = max(abs(np.max(self.u)), abs(np.min(self.u)), 10)
        self.ax3.set_ylim(-control_max*1.2, control_max*1.2)
        self.ax3.set_xlabel('Tiempo [s]')
        self.ax3.set_ylabel('Acción de control u')
        self.ax3.set_title('Acción de Control')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend()

        # Información del tiempo
        self.time_text = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes,
                                     fontsize=12, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

    def animate_frame(self, frame):
        """Función que actualiza la animación en cada frame"""
        # Actualizar péndulo
        cart_x = [self.cart_x[frame] - self.cart_width/2, self.cart_x[frame] + self.cart_width/2]
        cart_y = [0, 0]
        self.cart.set_data(cart_x, cart_y)

        pendulum_x = [self.cart_x[frame], self.pendulum_x[frame]]
        pendulum_y = [0, self.pendulum_y[frame]]
        self.pendulum_line.set_data(pendulum_x, pendulum_y)

        # Actualizar flechas de fuerza
        force_scale = 0.5  # escala para visualizar
        force_x = self.u[frame] * force_scale

        # Remover flechas anteriores (si existen)
        try:
            if hasattr(self, 'force_arrow') and self.force_arrow in self.ax1.patches:
                self.force_arrow.remove()
            if hasattr(self, 'disturbance_arrow') and self.disturbance_arrow in self.ax1.patches:
                self.disturbance_arrow.remove()
        except:
            pass  # Ignorar errores al remover flechas

        # Nueva flecha de control
        self.force_arrow = self.ax1.arrow(self.cart_x[frame], 0.2, force_x, 0,
                                        head_width=0.03, head_length=0.05,
                                        fc='green', ec='green', alpha=0.8)

        # Nueva flecha de perturbación
        dist_scale = 2.0
        dist_x = self.disturbance[frame] * dist_scale
        self.disturbance_arrow = self.ax1.arrow(self.pendulum_x[frame], self.pendulum_y[frame], dist_x, 0,
                                              head_width=0.02, head_length=0.03,
                                              fc='orange', ec='orange', alpha=0.6)

        # Actualizar gráficos de evolución
        current_t = self.t[:frame+1]
        current_theta = self.theta[:frame+1]
        current_ref = self.ref[:frame+1]
        current_u = self.u[:frame+1]
        current_dist = self.disturbance[:frame+1]

        self.theta_line.set_data(current_t, current_theta)
        self.ref_line.set_data(current_t, current_ref)
        self.control_line.set_data(current_t, current_u)
        self.disturbance_line.set_data(current_t, current_dist)

        # Actualizar texto de tiempo
        self.time_text.set_text('.2f')

        return [self.cart, self.pendulum_line, self.force_arrow, self.disturbance_arrow,
                self.theta_line, self.ref_line, self.control_line, self.disturbance_line, self.time_text]

    def create_animation(self, save_path=None, fps=30, bitrate=1800):
        """
        Crea la animación

        Args:
            save_path: Ruta para guardar el video (opcional)
            fps: Frames por segundo
            bitrate: Calidad del video
        """
        # Calcular frames a mostrar (uno cada pocos pasos para animación fluida)
        frame_step = max(1, int(self.dt * fps))
        frames = range(0, len(self.t), frame_step)

        print(f"Creando animación con {len(frames)} frames...")

        # Crear animación
        self.anim = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=frames,
            interval=1000/fps, blit=False, repeat=True  # Cambié blit=True a False por compatibilidad
        )

        # Guardar si se especifica
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:  # Solo crear directorio si no está vacío
                os.makedirs(save_dir, exist_ok=True)
            try:
                self.anim.save(save_path, writer='ffmpeg', fps=fps, bitrate=bitrate)
                print(f"Animación guardada en: {save_path}")
            except Exception as e:
                print(f"No se pudo guardar la animación (ffmpeg no disponible): {e}")
                print("Guardando frames individuales como alternativa...")

                # Guardar frames individuales
                self.save_frames(save_dir if save_dir else ".", fps=fps)

        return self.anim

    def save_frames(self, output_dir, fps=10):
        """
        Guarda frames individuales de la animación como imágenes

        Args:
            output_dir: Directorio donde guardar los frames
            fps: Frames por segundo (afecta cuántos frames guardar)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Frames a guardar (uno cada cierto intervalo)
        frame_step = max(1, int(1.0 / (self.dt * fps)))
        frames_to_save = range(0, len(self.t), frame_step)

        print(f"Guardando {len(frames_to_save)} frames en {output_dir}...")

        for i, frame_idx in enumerate(frames_to_save):
            # Actualizar animación al frame actual
            self.animate_frame(frame_idx)

            # Guardar frame
            frame_path = os.path.join(output_dir, "06d")
            self.fig.savefig(frame_path, dpi=150, bbox_inches='tight')

            if (i + 1) % 10 == 0:
                print(f"Guardado frame {i+1}/{len(frames_to_save)}")

        print(f"Frames guardados en: {output_dir}")
        print("Puede crear un GIF o video con herramientas externas usando estos frames")

    def show_animation(self):
        """Muestra la animación"""
        plt.show()

    def plot_evolution(self, save_path=None):
        """
        Crea un gráfico estático mostrando la evolución completa del sistema

        Args:
            save_path: Ruta para guardar el gráfico (opcional)
        """
        # Crear figura
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Gráfico 1: Evolución del ángulo
        ax1.plot(self.t, self.theta, 'b-', linewidth=2, label='θ(t)')
        ax1.plot(self.t, self.ref, 'k--', linewidth=1, label='Referencia')
        ax1.set_xlabel('Tiempo [s]')
        ax1.set_ylabel('Ángulo θ [rad]')
        ax1.set_title('Evolución del Ángulo del Péndulo')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Gráfico 2: Acción de control
        ax2.plot(self.t, self.u, 'r-', linewidth=2, label='u(t)')
        ax2.set_xlabel('Tiempo [s]')
        ax2.set_ylabel('Acción de control u')
        ax2.set_title('Acción del Controlador PID')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Gráfico 3: Perturbaciones
        ax3.plot(self.t, self.disturbance, color='orange', linewidth=2, label='Perturbación')
        ax3.set_xlabel('Tiempo [s]')
        ax3.set_ylabel('Perturbación')
        ax3.set_title('Perturbaciones Aplicadas')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Gráfico 4: Posición cartesiana del péndulo
        ax4.plot(self.pendulum_x, self.pendulum_y, 'r-', linewidth=1, alpha=0.7, label='Trayectoria')
        ax4.plot(self.pendulum_x[0], self.pendulum_y[0], 'go', markersize=8, label='Inicio')
        ax4.plot(self.pendulum_x[-1], self.pendulum_y[-1], 'ro', markersize=8, label='Fin')
        ax4.set_xlabel('Posición X [m]')
        ax4.set_ylabel('Posición Y [m]')
        ax4.set_title('Trayectoria del Extremo del Péndulo')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axis('equal')

        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:  # Solo crear directorio si no está vacío
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de evolución guardado en: {save_path}")

        return fig

# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    # Importar el controlador PID
    from respuesta_pid_pendulo import PIDController

    # Crear controlador PID
    pid = PIDController(Kp=80.5255, Ki=5.0000, Kd=10.0000)

    # Crear animador
    animator = PendulumAnimator(
        pendulum_model=None,  # El modelo se crea internamente
        controller=pid,
        t_total=15.0,
        dt=0.02,
        initial_theta=0.1,
        reference_func=lambda t: 0.0,  # Referencia constante en 0
        disturbance_func=lambda t: 0.01 * np.sin(2 * np.pi * 0.5 * t) if t > 5.0 else 0.0,  # Perturbación sinusoidal después de 5s
        cart_width=0.3,
        pendulum_length=0.5
    )

    # Crear gráfico de evolución
    print("\nCreando gráfico de evolución del sistema...")
    evolution_fig = animator.plot_evolution(save_path="pendulum_evolution.png")

    # Crear animación (guardar frames si ffmpeg no está disponible)
    print("\nCreando animación del péndulo...")
    print("Nota: Si ffmpeg no está instalado, se guardarán frames individuales")
    anim = animator.create_animation(save_path="pendulum_animation.mp4", fps=10)

    # Cerrar figura de evolución para evitar mostrarla
    plt.close(evolution_fig)

    print("\nAnimación y gráficos creados exitosamente!")
    print("Archivos guardados:")
    print("- pendulum_evolution.png: Gráfico de evolución del sistema")
    print("- pendulum_frames/: Carpeta con frames individuales de la animación")
    print("- pendulum_animation.mp4: Video de animación (si ffmpeg estaba disponible)")

    # No mostrar animación automáticamente para evitar bloqueo
    print("\nPara ver la animación, ejecute el script en un entorno con interfaz gráfica")
    print("o use los frames individuales para crear un video/GIF con herramientas externas")

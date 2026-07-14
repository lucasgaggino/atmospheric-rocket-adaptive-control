import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import os

def animate_from_csv(csv_path, output_video_path):
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    t = df['t'].values
    theta = df['y'].values
    u = df['u'].values
    
    # Extract RST controller coeffs and estimated ARX params
    s0 = df['s0'].values
    s1 = df['s1'].values
    s2 = df['s2'].values
    a1_est = df['a1_est'].values
    a2_est = df['a2_est'].values
    b1_est = df['b1_est'].values
    b2_est = df['b2_est'].values

    # Parameters for visualization
    cart_width = 0.4
    cart_height = 0.2
    pendulum_length = 0.6
    
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(3, 4)

    # Main Animation Subplot (Center)
    ax = fig.add_subplot(gs[:, 1:3])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Inverted Pendulum Animation\nFile: {os.path.basename(csv_path)}')
    ax.set_xlabel('Position [m]')
    ax.set_ylabel('Height [m]')

    # Angle Subplot (Left Top)
    ax_angle = fig.add_subplot(gs[0:2, 0])
    ax_angle.set_title('Angle (Degrees)')
    ax_angle.set_xlim(0, t[-1])
    theta_deg = np.degrees(theta)
    ax_angle.set_ylim(min(theta_deg) - 5, max(theta_deg) + 5)
    ax_angle.grid(True, alpha=0.3)
    line_angle, = ax_angle.plot([], [], 'b-', label='Theta [deg]', lw=1.5)
    ax_angle.legend(fontsize='small', loc='upper right')

    # Control Action Subplot (Left Bottom)
    ax_control = fig.add_subplot(gs[2, 0])
    ax_control.set_title('Control Action u (N)')
    ax_control.set_xlim(0, t[-1])
    ax_control.set_ylim(min(u) - 5, max(u) + 5)
    ax_control.grid(True, alpha=0.3)
    line_control, = ax_control.plot([], [], 'r-', label='u [N]', lw=1.5)
    ax_control.legend(fontsize='small', loc='upper right')
    ax_control.set_xlabel('Time [s]')

    # RST Controller Coeffs Subplot (Right Top)
    ax_rst = fig.add_subplot(gs[0, 3])
    ax_rst.set_title('RST S(q) coeffs')
    ax_rst.set_xlim(0, t[-1])
    # Auto-scale Y based on data range + margin
    s_min = min(s0.min(), s1.min(), s2.min())
    s_max = max(s0.max(), s1.max(), s2.max())
    margin = max(5.0, 0.05 * (s_max - s_min))
    ax_rst.set_ylim(s_min - margin, s_max + margin)
    ax_rst.grid(True, alpha=0.3)
    
    line_s0, = ax_rst.plot([], [], 'r-', label='s0', lw=1.5)
    line_s1, = ax_rst.plot([], [], 'g-', label='s1', lw=1.5)
    line_s2, = ax_rst.plot([], [], 'b-', label='s2', lw=1.5)
    ax_rst.legend(fontsize='small', loc='upper right')

    # Estimated Params Subplot 1 (Right Middle)
    ax_est_a = fig.add_subplot(gs[1, 3])
    ax_est_a.set_title('Estimated a1, a2')
    ax_est_a.set_xlim(0, t[-1])
    
    a_min = min(a1_est.min(), a2_est.min())
    a_max = max(a1_est.max(), a2_est.max())
    ax_est_a.set_ylim(a_min - 0.5, a_max + 0.5)
    ax_est_a.grid(True, alpha=0.3)

    line_a1, = ax_est_a.plot([], [], 'c-', label='a1', lw=1.5)
    line_a2, = ax_est_a.plot([], [], 'm-', label='a2', lw=1.5)
    ax_est_a.legend(fontsize='small', loc='upper right')

    # Estimated Params Subplot 2 (Right Bottom)
    ax_est_b = fig.add_subplot(gs[2, 3])
    ax_est_b.set_title('Estimated b1, b2')
    ax_est_b.set_xlim(0, t[-1])

    b_min = min(b1_est.min(), b2_est.min())
    b_max = max(b1_est.max(), b2_est.max())
    ax_est_b.set_ylim(b_min - 0.1, b_max + 0.1)
    ax_est_b.grid(True, alpha=0.3)

    line_b1, = ax_est_b.plot([], [], 'y-', label='b1', lw=1.5)
    line_b2, = ax_est_b.plot([], [], 'k-', label='b2', lw=1.5)
    ax_est_b.legend(fontsize='small', loc='upper right')
    ax_est_b.set_xlabel('Time [s]')

    # Elements
    # Cart (Black box)
    # Centered at (0,0) initially. 
    # Rectangle defined by bottom-left corner.
    cart = patches.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, 
                             fc='black', ec='black', label='Cart')
    ax.add_patch(cart)

    # Pendulum (Line + Mass)
    # Rod
    line, = ax.plot([], [], 'o-', lw=3, color='black', markersize=0) # rod line
    # Mass
    mass_radius = 0.05
    mass = patches.Circle((0, 0), radius=mass_radius, fc='red', ec='black', zorder=10)
    ax.add_patch(mass)

    # Control Arrow (Red arrow)
    arrow_ax = ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.05, fc='red', ec='red', visible=False)

    # Time text
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    def init():
        cart.set_xy((-cart_width/2, -cart_height/2))
        line.set_data([], [])
        mass.center = (0, 0)
        time_text.set_text('')
        
        # Init lines for subplots
        line_s0.set_data([], [])
        line_s1.set_data([], [])
        line_s2.set_data([], [])
        line_a1.set_data([], [])
        line_a2.set_data([], [])
        line_b1.set_data([], [])
        line_b2.set_data([], [])
        line_angle.set_data([], [])
        line_control.set_data([], [])
        
        return cart, line, mass, time_text, line_s0, line_s1, line_s2, line_a1, line_a2, line_b1, line_b2, line_angle, line_control

    def update(frame):
        current_theta = theta[frame]
        current_u = u[frame]
        current_t = t[frame]
        
        # Update Subplots Lines
        # For efficiency, we update data up to current frame
        # Since we are stepping frames, using slice :frame+1 is correct
        t_slice = t[:frame+1]
        
        line_s0.set_data(t_slice, s0[:frame+1])
        line_s1.set_data(t_slice, s1[:frame+1])
        line_s2.set_data(t_slice, s2[:frame+1])
        
        line_a1.set_data(t_slice, a1_est[:frame+1])
        line_a2.set_data(t_slice, a2_est[:frame+1])
        
        line_b1.set_data(t_slice, b1_est[:frame+1])
        line_b2.set_data(t_slice, b2_est[:frame+1])
        
        # Update Angle (deg) and Control (N)
        theta_slice = theta[:frame+1]
        theta_deg_slice = np.degrees(theta_slice)
        line_angle.set_data(t_slice, theta_deg_slice)
        
        u_slice = u[:frame+1]
        line_control.set_data(t_slice, u_slice)

        # Cart is fixed at x=0 for visualization (as we don't have x data)
        cart_x = 0
        cart_y = 0
        
        # Pendulum position
        # theta = 0 is UP (unstable equilibrium in the simulation context provided previously)
        # x_p = x_c + L * sin(theta)
        # y_p = y_c + L * cos(theta)
        # Note: In standard mathematical derivation for inverted pendulum, theta=0 is often UP.
        # Let's verify with the CSV data behavior. 
        # Initial condition was 0.1 rad (approx 5.7 deg). 
        # If theta=0 is UP, then pendulum is slightly tilted.
        
        pend_x = cart_x + pendulum_length * np.sin(current_theta)
        pend_y = cart_y + pendulum_length * np.cos(current_theta)
        
        line.set_data([cart_x, pend_x], [cart_y, pend_y])
        mass.center = (pend_x, pend_y)
        
        # Control Arrow
        # Originates from cart center (or side).
        # Direction based on sign of u.
        # Scale u for visualization
        u_scale = 0.02 # Adjustment factor for visualization
        arrow_len = abs(current_u) * u_scale
        
        # Limit max arrow length for clean visualization
        max_arrow_len = 1.0
        arrow_len = min(arrow_len, max_arrow_len)

        nonlocal arrow_ax
        if arrow_ax:
            arrow_ax.remove()
        
        if abs(current_u) > 0.1: # Threshold to show arrow
            sign = np.sign(current_u)
            # Arrow points in the direction of force/acceleration?
            # Typically u is force on cart. If u > 0, force is to the right.
            # Start of arrow
            arrow_start_x = cart_x
            arrow_start_y = cart_y 
            
            # If we want the arrow to point in direction of force:
            dx = sign * arrow_len
            dy = 0
            
            # To make it look like "pushing" or "pulling"?
            # Standard: arrow originating from center pointing in direction of force.
            arrow_ax = ax.arrow(arrow_start_x, arrow_start_y, dx, dy, 
                                head_width=0.08, head_length=0.1, 
                                fc='red', ec='red', length_includes_head=True, width=0.02)
        else:
            # Create invisible arrow to keep reference valid if needed, or just None
            arrow_ax = ax.arrow(0,0,0,0, visible=False)

        time_text.set_text(f'Time = {current_t:.2f} s\nTheta = {current_theta:.2f} rad\nu = {current_u:.2f}')
        
        return cart, line, mass, arrow_ax, time_text, line_s0, line_s1, line_s2, line_a1, line_a2, line_b1, line_b2, line_angle, line_control

    # Frames
    # reduce fps for speed if needed, or skip frames
    # Data is Ts = 0.02s (50 Hz). 
    # Video at 30fps. 
    # We can use every frame or skip. Let's use every 2nd frame to speed up generation/reduce size if it's long.
    # Total time 20s = 1000 frames. 1000 frames is fine.
    
    fps = 30
    step = 2 # Process every 2nd data point (approx 0.04s per frame -> 25fps real time playback speed approx)
    frames = range(0, len(t), step)
    
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=1000/fps)
    
    # Create progress callback
    def progress_callback(current, total):
        if current % 10 == 0:
            print(f"Processing frame {current}/{total}", end='\r')

    # Save
    print(f"Saving animation to {output_video_path}...")
    try:
        ani.save(output_video_path, writer='ffmpeg', fps=fps, progress_callback=progress_callback)
        print("\nDone.")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Trying 'pillow' writer (GIF)...")
        gif_path = output_video_path.replace('.mp4', '.gif')
        ani.save(gif_path, writer='pillow', fps=fps)
        print(f"Saved as GIF: {gif_path}")

if __name__ == "__main__":
    csv_file = "saved_runs_pid_autotunning/pert_escalon.csv"
    output_file = "saved_runs_pid_autotunning/pert_escalon_video.mp4"
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    animate_from_csv(csv_file, output_file)
    
    csv_file = "saved_runs_pid_autotunning/pert_sinusoidal.csv"
    output_file = "saved_runs_pid_autotunning/pert_sinusoidal_video.mp4"
    animate_from_csv(csv_file, output_file)
    
    csv_file = "saved_runs_pid_autotunning/sin_perturbaciones.csv"
    output_file = "saved_runs_pid_autotunning/sin_perturbaciones_video.mp4"
    animate_from_csv(csv_file, output_file)
    
    csv_file = "saved_runs_pid_autotunning/ruido_medicion.csv"
    output_file = "saved_runs_pid_autotunning/ruido_medicion_video.mp4"
    animate_from_csv(csv_file, output_file)
    
    
    


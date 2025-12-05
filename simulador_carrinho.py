import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os

# ==============================================================================
# 1. CLASSE DO CONTROLADOR PID
# ==============================================================================
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.min_out, self.max_out = output_limits
        self.integral = 0.0
        self.prev_error = 0.0

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        P = self.Kp * error
        I = self.Ki * self.integral
        D = self.Kd * ((error - self.prev_error) / dt) if dt > 0 else 0.0
        self.prev_error = error
        output = P + I + D
        if self.min_out is not None and self.max_out is not None:
            output = np.clip(output, self.min_out, self.max_out)
        return output

# ==============================================================================
# 2. CLASSE DO ROVER 4X4 (MOTORES INDEPENDENTES)
# ==============================================================================
class FourWheelRover:
    """
    Rover com 4 motores independentes.
    Dynamics Inputs: [v_FL, v_FR, v_RL, v_RR]
    """
    def __init__(self):
        self.width = 0.3      # Distância lateral entre rodas (m)
        self.wheel_radius = 0.05 
        self.mass = 2.0       
        
        self.max_voltage = 6.0
        self.motor_kv = 5.0   # rad/s por Volt
        
        # Estado [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0]) 

    def dynamics(self, t, current_state, inputs):
        """
        inputs: array de 4 voltagens [V_FL, V_FR, V_RL, V_RR]
        FL: Front-Left, FR: Front-Right, RL: Rear-Left, RR: Rear-Right
        """
        x, y, theta = current_state
        
        # 1. Saturação individual
        volts = np.clip(inputs, -self.max_voltage, self.max_voltage)
        v_fl, v_fr, v_rl, v_rr = volts
        
        # 2. Conversão para velocidade angular da roda (w = V * kv)
        # E depois para velocidade linear da roda (v = w * r)
        # Fator r*kv
        factor = self.wheel_radius * self.motor_kv
        
        vel_fl = v_fl * factor
        vel_fr = v_fr * factor
        vel_rl = v_rl * factor
        vel_rr = v_rr * factor
        
        # 3. Cinemática 4WD Skid-Steer
        # Velocidade do lado esquerdo (média das rodas esquerdas)
        vel_left_side = (vel_fl + vel_rl) / 2.0
        
        # Velocidade do lado direito (média das rodas direitas)
        vel_right_side = (vel_fr + vel_rr) / 2.0
        
        # Velocidade Linear do Robô (Média total)
        v_linear = (vel_left_side + vel_right_side) / 2.0
        
        # Velocidade Angular (Diferença / Largura)
        omega = (vel_right_side - vel_left_side) / self.width
        
        # 4. Equações no mundo
        dx = v_linear * np.cos(theta)
        dy = v_linear * np.sin(theta)
        dtheta = omega
        
        return np.array([dx, dy, dtheta]), v_linear, omega, volts

# ==============================================================================
# 3. SIMULAÇÃO
# ==============================================================================
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def run_rover_simulation(rover, mode, target, duration, dt=0.05):
    
    # PIDs (Usados apenas nos modos 1 e 2)
    pid_lin = PIDController(Kp=4.0, Ki=0.5, Kd=0.1, output_limits=(-6, 6))
    pid_ang = PIDController(Kp=15.0, Ki=0.1, Kd=0.5, output_limits=(-6, 6))
    
    state = rover.state.copy()
    time = 0.0
    
    # Gerenciamento de Alvos (Modo Position)
    current_target_idx = 0
    targets_list = []
    if mode == 'position':
        if isinstance(target[0], (int, float)): targets_list = [target]
        else: targets_list = target
    
    # Histórico estendido para 4 motores
    history = {
        'time': [], 'x': [], 'y': [], 'theta': [], 
        'v_linear': [], 'omega': [], 
        'v_fl': [], 'v_fr': [], 'v_rl': [], 'v_rr': []
    }
    
    steps = int(duration / dt)
    print(f"Iniciando simulação Modo: {mode.upper()}...")

    for i in range(steps):
        x, y, theta = state
        
        # Inicializa comandos para as 4 rodas
        u_fl = u_fr = u_rl = u_rr = 0.0
        
        if mode == 'single_wheel':
            # MODO 3: Apenas Roda Dianteira Esquerda
            # Tensão fixa definida pelo usuário (target é apenas um valor float de tensão)
            voltage_input = target
            u_fl = voltage_input
            u_fr = 0.0
            u_rl = 0.0
            u_rr = 0.0
            
        elif mode == 'position':
            # Lógica PID existente (Modo 1)
            linear_cmd = 0.0
            turn_cmd = 0.0
            
            if current_target_idx < len(targets_list):
                tx, ty = targets_list[current_target_idx]
                dx = tx - x
                dy = ty - y
                dist_error = np.sqrt(dx**2 + dy**2)
                
                if dist_error < 0.1:
                    current_target_idx += 1
                    pid_lin.reset()
                    pid_ang.reset()
                else:
                    target_heading = np.arctan2(dy, dx)
                    heading_error = normalize_angle(target_heading - theta)
                    speed_factor = max(0, 1 - (abs(heading_error) / (np.pi/2)))
                    
                    pid_lin.set_setpoint(0.0) 
                    linear_voltage = np.clip(pid_lin.Kp * dist_error, -6, 6)
                    linear_cmd = linear_voltage * speed_factor
                    
                    pid_ang.set_setpoint(0.0)
                    turn_cmd = pid_ang.update(-heading_error, dt)
            
            # Distribui comandos do PID para as 4 rodas
            # Esquerda = Base - Giro
            u_fl = u_rl = linear_cmd - turn_cmd
            # Direita = Base + Giro
            u_fr = u_rr = linear_cmd + turn_cmd

        elif mode == 'velocity':
            # Lógica PID existente (Modo 2)
            v_des, theta_des = target
            ff_voltage = v_des / (rover.wheel_radius * rover.motor_kv * 0.25) # Ajustado para 4 rodas
            linear_cmd = np.clip(ff_voltage, -6, 6)
            
            heading_error = normalize_angle(theta_des - theta)
            pid_ang.set_setpoint(0.0)
            turn_cmd = pid_ang.update(-heading_error, dt)
            
            u_fl = u_rl = linear_cmd - turn_cmd
            u_fr = u_rr = linear_cmd + turn_cmd

        # --- Dinâmica ---
        inputs_4wd = [u_fl, u_fr, u_rl, u_rr]
        state_dot, v_real, w_real, applied_volts = rover.dynamics(time, state, inputs_4wd)
        
        state += state_dot * dt
        state[2] = normalize_angle(state[2])
        time += dt
        
        # Salvar dados
        history['time'].append(time)
        history['x'].append(state[0])
        history['y'].append(state[1])
        history['theta'].append(state[2])
        history['v_linear'].append(v_real)
        history['omega'].append(w_real)
        history['v_fl'].append(applied_volts[0])
        history['v_fr'].append(applied_volts[1])
        history['v_rl'].append(applied_volts[2])
        history['v_rr'].append(applied_volts[3])

    return history

# ==============================================================================
# 4. VISUALIZAÇÃO
# ==============================================================================
def create_visuals(history, target, mode, filename_suffix):
    t = np.array(history['time'])
    x = np.array(history['x'])
    y = np.array(history['y'])
    theta = np.array(history['theta'])
    
    # --- Gráficos Estáticos ---
    fig_plot, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig_plot.suptitle(f'Telemetria Rover 4WD - Modo: {mode.upper()}', fontsize=16)
    
    # 1. Trajetória
    axs[0, 0].plot(x, y, 'b-', label='Caminho')
    if mode == 'position':
        targets_plot = target if isinstance(target[0], (list, tuple)) else [target]
        for i, pt in enumerate(targets_plot):
            axs[0, 0].plot(pt[0], pt[1], 'rx', markersize=10, label=f'Alvo {i+1}')
    elif mode == 'single_wheel':
         axs[0, 0].plot(x[0], y[0], 'go', label='Início')
         
    axs[0, 0].set_title('Trajetória XY')
    axs[0, 0].set_xlabel('X (m)')
    axs[0, 0].set_ylabel('Y (m)')
    axs[0, 0].legend()
    axs[0, 0].axis('equal')
    axs[0, 0].grid(True)

    # 2. Heading
    axs[0, 1].plot(t, np.degrees(theta), 'g')
    axs[0, 1].set_title('Heading (Graus)')
    axs[0, 1].grid(True)
    
    # 3. Tensões Individuais (Mostrando as 4)
    axs[1, 0].plot(t, history['v_fl'], label='FL', alpha=0.7)
    axs[1, 0].plot(t, history['v_fr'], label='FR', alpha=0.7, linestyle='--')
    axs[1, 0].plot(t, history['v_rl'], label='RL', alpha=0.5, linestyle=':')
    axs[1, 0].plot(t, history['v_rr'], label='RR', alpha=0.5, linestyle='-.')
    axs[1, 0].set_title('Tensão nos 4 Motores (V)')
    axs[1, 0].legend(loc='upper right', fontsize='small', ncol=2)
    axs[1, 0].grid(True)
    
    # 4. Velocidades
    axs[1, 1].plot(t, history['v_linear'], label='Linear (m/s)')
    axs[1, 1].plot(t, history['omega'], label='Angular (rad/s)', linestyle='--')
    axs[1, 1].set_title('Velocidades do Corpo')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # --- Animação GIF ---
    fig_anim, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min(x)-1, max(x)+1)
    ax.set_ylim(min(y)-1, max(y)+1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"Simulação 4WD: {mode.upper()}")
    
    trail, = ax.plot([], [], 'b:', linewidth=1)
    w_rob, h_rob = 0.4, 0.3
    
    # Corpo e Rodas
    robot_rect = patches.Rectangle((0,0), w_rob, h_rob, fc='orange', ec='black')
    ax.add_patch(robot_rect)
    
    wheels = []
    colors = ['red', 'black', 'red', 'black'] # FL(Vermelho para destaque), FR, RL, RR
    for c in colors:
        w = patches.Rectangle((0,0), 0.1, 0.05, fc=c)
        ax.add_patch(w)
        wheels.append(w)

    def init():
        robot_rect.set_xy((-w_rob/2, -h_rob/2))
        trail.set_data([], [])
        return [robot_rect, trail] + wheels

    def update(frame):
        idx = frame * 3
        if idx >= len(x): idx = len(x)-1
        
        cx, cy, cth = x[idx], y[idx], theta[idx]
        trail.set_data(x[:idx], y[:idx])
        
        deg = np.degrees(cth)
        rot_mat = np.array([[np.cos(cth), -np.sin(cth)], [np.sin(cth), np.cos(cth)]])
        
        # Corpo
        corner = rot_mat @ np.array([-w_rob/2, -h_rob/2])
        robot_rect.set_xy((cx + corner[0], cy + corner[1]))
        robot_rect.angle = deg
        
        # Rodas: FL, FR, RL, RR
        offsets = [[0.1, 0.15], [0.1, -0.15], [-0.1, 0.15], [-0.1, -0.15]]
        
        # Se estiver no modo single wheel, destaca a roda FL mudando a cor/tamanho se quiser
        # Aqui apenas movemos
        for i, w_patch in enumerate(wheels):
            pos_local = np.array(offsets[i])
            pos_world = (rot_mat @ pos_local) + np.array([cx, cy])
            w_corner_local = np.array([-0.05, -0.025])
            final_corner = pos_world + (rot_mat @ w_corner_local)
            
            w_patch.set_xy(final_corner)
            w_patch.angle = deg

        return [robot_rect, trail] + wheels

    frames = len(t) // 3
    ani = FuncAnimation(fig_anim, update, frames=frames, init_func=init, blit=True)
    
    gif_name = f"rover_4wd_{filename_suffix}.gif"
    print(f"Gerando GIF: {gif_name} ...")
    ani.save(gif_name, writer='pillow', fps=30)
    print(f"GIF salvo em: {os.path.abspath(gif_name)}")
    plt.close(fig_anim)

# ==============================================================================
# 5. MAIN
# ==============================================================================
if __name__ == "__main__":
    print("\n=== Simulador Rover 4WD (Motores Independentes) ===")
    print("1. Navegar Caminho (Setpoints)")
    print("2. Controle Velocidade/Atitude")
    print("3. Teste Motor Único (Dianteiro Esquerdo)")
    
    try:
        escolha = input("Opção: ").strip()
        rover = FourWheelRover()
        
        if escolha == '1':
            print("Digite 2 pontos de parada.")
            p1 = [float(input("X1: ")), float(input("Y1: "))]
            p2 = [float(input("X2: ")), float(input("Y2: "))]
            dur = float(input("Duração (s) [15]: ") or 15)
            
            hist = run_rover_simulation(rover, 'position', [p1, p2], dur)
            create_visuals(hist, [p1, p2], 'position', 'path')

        elif escolha == '2':
            v = float(input("Velocidade (m/s): "))
            ang = float(input("Ângulo (graus): "))
            dur = float(input("Duração (s) [5]: ") or 5)
            
            hist = run_rover_simulation(rover, 'velocity', [v, np.radians(ang)], dur)
            create_visuals(hist, [v, ang], 'velocity', 'control')

        elif escolha == '3':
            print("\n--- Teste de Motor Único (FL) ---")
            print("As outras 3 rodas ficarão paradas (V=0).")
            volts = float(input("Tensão na roda FL (0 a 6V): "))
            dur = float(input("Duração (s) [10]: ") or 10)
            
            # Passamos a tensão como 'target'
            hist = run_rover_simulation(rover, 'single_wheel', volts, dur)
            create_visuals(hist, volts, 'single_wheel', 'motor_FL_only')
            
        else:
            print("Opção inválida.")

    except ValueError:
        print("Erro de entrada numérica.")
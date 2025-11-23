import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# ==============================================================================
# 1. CLASSE DO CONTROLADOR PID (Sem mudanças)
# ==============================================================================
class PIDController:
    """
    Implementação de um controlador PID (Proporcional-Integral-Derivativo).
    """
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, anti_windup_limit=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.anti_windup_limit = anti_windup_limit

    def set_setpoint(self, setpoint):
        """Define o valor desejado (alvo)."""
        self.setpoint = setpoint

    def update(self, current_value, dt):
        """Calcula a saída do controlador com base no valor atual."""
        error = self.setpoint - current_value
        
        P_term = self.Kp * error
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.anti_windup_limit, self.anti_windup_limit)
        I_term = self.Ki * self.integral
        
        if dt > 1e-6:
            derivative = (error - self.prev_error) / dt
            D_term = self.Kd * derivative
        else:
            D_term = 0.0 # Evita divisão por zero
            
        self.prev_error = error
        output = P_term + I_term + D_term
        return output

# ==============================================================================
# 2. CLASSE DO QUADRICÓPTERO (Sem mudanças)
# ==============================================================================
class Quadrotor:
    """
    Classe que representa o modelo dinâmico de um quadricóptero.
    """
    def __init__(self):
        # Parâmetros físicos (OS4)
        self.m = 0.650
        self.g = 9.81
        self.Ixx = 7.5e-3
        self.Iyy = 7.5e-3
        self.Izz = 1.3e-2
        self.l = 0.23
        self.b = 3.13e-5 # Coef. de empuxo
        self.d = 7.5e-7  # Coef. de arrasto
        self.ground_k = 500.0
        self.ground_c = 100.0
        self.initial_state = np.zeros(12)

    def dynamics(self, t, state, inputs):
        """
        Define as equações diferenciais.
        """
        w1, w2, w3, w4 = inputs
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        # Forças e Torques
        T = self.b * (w1**2 + w2**2 + w3**2 + w4**2)
        tau_phi = self.l * self.b * (w4**2 - w2**2)
        tau_theta = self.l * self.b * (w1**2 - w3**2)
        tau_psi = self.d * (w1**2 - w2**2 + w3**2 - w4**2)
        
        # Matriz de Rotação
        R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        R = R_z @ R_y @ R_x
        
        # Aceleração translacional
        thrust_world = R @ np.array([0, 0, T])
        gravity_world = np.array([0, 0, -self.m * self.g])
        force_total = thrust_world + gravity_world

        # Força de contato com o solo
        force_ground = 0
        if z < 0:
            force_ground = -self.ground_k * z - self.ground_c * vz
        force_total[2] += force_ground

        acceleration = force_total / self.m
        ax, ay, az = acceleration

        # Dinâmica rotacional
        p_dot = (tau_phi - q * r * (self.Izz - self.Iyy)) / self.Ixx
        q_dot = (tau_theta - p * r * (self.Ixx - self.Izz)) / self.Iyy
        r_dot = (tau_psi - p * q * (self.Iyy - self.Ixx)) / self.Izz
        
        # Conversão de taxas
        T_euler = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        
        if abs(np.cos(theta)) < 1e-6:
            euler_rates = np.zeros(3)
        else:
            euler_rates = T_euler @ np.array([p, q, r])

        phi_dot = euler_rates[0]
        theta_dot = euler_rates[1]
        psi_dot = euler_rates[2]

        return [vx, vy, vz, ax, ay, az, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]

# ==============================================================================
# 3. FUNÇÃO DE SIMULAÇÃO (MODIFICADA PARA VELOCIDADE E MODO)
# ==============================================================================
def run_pid_simulation(quad, targets, duration, gain_dicts, control_mode='position', dt=0.01, verbose=False):
    """
    Executa a simulação de controle PID em loop fechado.
    AGORA SUPORTA 'control_mode' ('position' ou 'velocity')
    e um dicionário 'targets'.
    """
    
    # --- Ponto 1: Ganhos PID ---
    z_gains = gain_dicts['z']
    attitude_gains = gain_dicts['attitude']
    yaw_gains = gain_dicts['yaw']

    # --- Ponto 2: Inicialização dos Controladores ---
    pid_z = PIDController(setpoint=targets['z'], **z_gains)
    pid_psi = PIDController(setpoint=targets['psi'], **yaw_gains)
    pid_phi = PIDController(**attitude_gains) # Atitude (loop interno)
    pid_theta = PIDController(**attitude_gains) # Atitude (loop interno)

    # --- NOVO: Controladores de Modo Específico (Loop Externo X/Y) ---
    if control_mode == 'position':
        xy_gains = gain_dicts['position_xy']
        pid_x_or_vx = PIDController(setpoint=targets['x'], **xy_gains)
        pid_y_or_vy = PIDController(setpoint=targets['y'], **xy_gains)
        if verbose: print(f"Modo de controle: POSIÇÃO. Alvo: ({targets['x']}, {targets['y']}, {targets['z']})")
            
    elif control_mode == 'velocity':
        vel_xy_gains = gain_dicts['velocity_xy']
        pid_x_or_vx = PIDController(setpoint=targets['vx'], **vel_xy_gains)
        pid_y_or_vy = PIDController(setpoint=targets['vy'], **vel_xy_gains)
        if verbose: print(f"Modo de controle: VELOCIDADE. Alvo: ({targets['vx']} m/s, {targets['vy']} m/s)")
            
    else:
        raise ValueError("Modo de controle desconhecido. Use 'position' ou 'velocity'.")

    # --- Ponto 3: Misturador de Controle (Sem mudanças) ---
    b, d, l = quad.b, quad.d, quad.l
    alloc_matrix = np.array([[b, b, b, b],
                             [0, -l*b, 0, l*b],
                             [l*b, 0, -l*b, 0],
                             [d, -d, d, -d]])
    inv_alloc_matrix = np.linalg.inv(alloc_matrix)
    max_thrust_per_motor = (quad.m * quad.g * 2) / 4.0
    max_w_sq = max_thrust_per_motor / quad.b

    # --- Ponto 4: O Loop de Simulação ---
    state = quad.initial_state.copy()
    time = 0.0
    times = [time]
    states = [state]
    num_steps = int(duration / dt)
    
    for i in range(num_steps):
        # Desempacota o estado atual
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        # 1. Loop Externo (Z e Yaw são sempre iguais)
        thrust_correction = pid_z.update(z, dt)
        total_thrust = (quad.m * quad.g) + thrust_correction
        psi_des = targets['psi'] # Mantém o setpoint de yaw
        
        # --- NOVO: Lógica de Controle de Modo ---
        if control_mode == 'position':
            # Loop Externo (Posição) -> Saída é Ângulo Desejado
            # O controlador calcula o erro (target_x - x)
            theta_des = pid_x_or_vx.update(x, dt)
            phi_des = -pid_y_or_vy.update(y, dt) # Nota: Y-control é -phi
        
        elif control_mode == 'velocity':
            # Loop Externo (Velocidade) -> Saída é Ângulo Desejado
            # O controlador calcula o erro (target_vx - vx)
            theta_des = pid_x_or_vx.update(vx, dt)
            phi_des = -pid_y_or_vy.update(vy, dt) # Nota: Y-control é -phi

        # 2. Limites de Atitude (Clipping)
        max_angle = np.deg2rad(30)
        phi_des = np.clip(phi_des, -max_angle, max_angle)
        theta_des = np.clip(theta_des, -max_angle, max_angle)

        # 3. Loop Interno (Atitude)
        pid_phi.set_setpoint(phi_des)
        pid_theta.set_setpoint(theta_des)
        pid_psi.set_setpoint(psi_des)
        
        # PIDs internos leem o erro (phi_des - phi) e geram TORQUES
        tau_phi = pid_phi.update(phi, dt)
        tau_theta = pid_theta.update(theta, dt)
        tau_psi = pid_psi.update(psi, dt)

        # 4. Misturador
        control_vector = np.array([total_thrust, tau_phi, tau_theta, tau_psi])
        w_squared = inv_alloc_matrix @ control_vector
        w_squared = np.clip(w_squared, 0, max_w_sq)
        inputs = np.sqrt(w_squared)
        
        # 5. Atualização da Dinâmica
        state_dot = quad.dynamics(time, state, inputs)
        state = state + np.array(state_dot) * dt
        
        # 6. Armazenamento
        time += dt
        times.append(time)
        states.append(state.copy())

    if verbose:
        print("Simulação concluída.")
    
    solution = type('Solution', (object,), {})()
    solution.t = np.array(times)
    solution.y = np.array(states).T
    
    return solution

# ==============================================================================
# 4. FUNÇÃO DE ANIMAÇÃO (MODIFICADA)
# ==============================================================================
def animate_and_save_3d(sol, movement_type, duration, quad_params, plot_target_marker):
    """
    Anima o resultado.
    'plot_target_marker' é um [x,y,z] usado apenas para desenhar o alvo no gráfico.
    """
    x_data, y_data, z_data = sol.y[0], sol.y[1], sol.y[2]
    phi_data, theta_data, psi_data = sol.y[6], sol.y[7], sol.y[8]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calcula limites para garantir que a trajetória e o alvo estejam visíveis
    all_x = np.append(x_data, [0, plot_target_marker[0]])
    all_y = np.append(y_data, [0, plot_target_marker[1]])
    all_z = np.append(z_data, [0, plot_target_marker[2]])

    max_range = max(np.max(all_x) - np.min(all_x), 
                    np.max(all_y) - np.min(all_y), 
                    np.max(all_z) - np.min(all_z), 
                    1.0) # Garante zoom mínimo
    
    mid_x = (np.max(all_x) + np.min(all_x)) / 2
    mid_y = (np.max(all_y) + np.min(all_y)) / 2
    mid_z = (np.max(all_z) + np.min(all_z)) / 2
    
    ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
    ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
    ax.set_zlim(0, mid_z + max_range * 0.6) # Z começa do chão

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Animação 3D: {movement_type.replace("_", " ").title()}', fontsize=16)

    # Chão
    X_ground, Y_ground = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10), 
                                     np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10))
    Z_ground = np.zeros_like(X_ground)
    ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='gray')

    trail, = ax.plot([], [], [], '-', lw=2, color='blue', label='Rastro da Trajetória')
    line1, = ax.plot([], [], [], '-', lw=3, color='red') 
    line2, = ax.plot([], [], [], '-', lw=3, color='red', label='Quadricóptero (X)')
    
    # Adiciona o ponto de alvo (visualização)
    ax.plot([plot_target_marker[0]], [plot_target_marker[1]], [plot_target_marker[2]], 
            'go', markersize=10, label='Alvo/Destino Projetado')
    ax.legend(loc='upper left')

    arm_length = quad_params.l 
    p_fl = np.array([arm_length, arm_length, 0]) / np.sqrt(2)
    p_fr = np.array([arm_length, -arm_length, 0]) / np.sqrt(2)
    p_rl = np.array([-arm_length, arm_length, 0]) / np.sqrt(2)
    p_rr = np.array([-arm_length, -arm_length, 0]) / np.sqrt(2)

    def init():
        trail.set_data_3d([], [], [])
        line1.set_data_3d([], [], [])
        line2.set_data_3d([], [], [])
        return trail, line1, line2

    def update(frame):
        idx = frame * 5 
        if idx >= len(x_data):
            idx = len(x_data) - 1
            
        x_c, y_c, z_c = x_data[idx], y_data[idx], z_data[idx]
        phi, theta, psi = phi_data[idx], theta_data[idx], psi_data[idx]

        R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        R = R_z @ R_y @ R_x
        
        p_fl_w = (R @ p_fl) + np.array([x_c, y_c, z_c])
        p_fr_w = (R @ p_fr) + np.array([x_c, y_c, z_c])
        p_rl_w = (R @ p_rl) + np.array([x_c, y_c, z_c])
        p_rr_w = (R @ p_rr) + np.array([x_c, y_c, z_c])
        
        line1.set_data_3d([p_fl_w[0], p_rr_w[0]], [p_fl_w[1], p_rr_w[1]], [p_fl_w[2], p_rr_w[2]])
        line2.set_data_3d([p_fr_w[0], p_rl_w[0]], [p_fr_w[1], p_rl_w[1]], [p_fr_w[2], p_rl_w[2]])
        
        trail.set_data_3d(x_data[:idx+1], y_data[:idx+1], z_data[:idx+1])
        return trail, line1, line2

    num_frames_anim = len(sol.t) // 5
    fps = int(num_frames_anim / duration)
    fps = np.clip(fps, 15, 60)

    ani = FuncAnimation(fig, update, frames=num_frames_anim, init_func=init, blit=True)
    
    file_name = f'animacao_3d_{movement_type}.gif'
    print(f'Criando animação: {file_name}...')
    ani.save(file_name, writer='pillow', fps=fps)
    print(f'Animação salva como: {os.path.abspath(file_name)}')
    plt.close(fig)

# ==============================================================================
# 5. NOVAS FUNÇÕES: PLOTAGEM DE RESPOSTA DE VELOCIDADE
# ==============================================================================
def plot_velocity_responses(sol, targets, file_tag):
    """
    Cria e salva gráficos de linha para as respostas dos eixos X, Y e Z.
    """
    print(f"Gerando gráficos de resposta para: {file_tag}...")
    
    t = sol.t
    # Posição
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    # Velocidade
    vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]
    # Atitude
    phi, theta, psi = sol.y[6], sol.y[7], sol.y[8]

    target_vx, target_vy = targets['vx'], targets['vy']
    target_z, target_psi = targets['z'], targets['psi']

    fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f'Respostas do Controlador - Modo Velocidade - {file_tag.replace("_", " ").title()}', fontsize=16)

    # --- Velocidade X ---
    axs[0].plot(t, vx, 'b-', label='Velocidade X Real (vx)')
    axs[0].axhline(y=target_vx, color='r', linestyle='--', label=f'Alvo vx ({target_vx:.2f} m/s)')
    axs[0].set_ylabel('Velocidade X (m/s)')
    axs[0].legend()
    axs[0].grid(True)

    # --- Velocidade Y ---
    axs[1].plot(t, vy, 'b-', label='Velocidade Y Real (vy)')
    axs[1].axhline(y=target_vy, color='r', linestyle='--', label=f'Alvo vy ({target_vy:.2f} m/s)')
    axs[1].set_ylabel('Velocidade Y (m/s)')
    axs[1].legend()
    axs[1].grid(True)

    # --- Posição Z (Altitude) ---
    axs[2].plot(t, z, 'b-', label='Altitude Real (z)')
    axs[2].axhline(y=target_z, color='r', linestyle='--', label=f'Alvo z ({target_z:.2f} m)')
    axs[2].set_ylabel('Altitude (m)')
    axs[2].legend()
    axs[2].grid(True)
    
    # --- Atitude PSI (Yaw) ---
    axs[3].plot(t, np.rad2deg(psi), 'b-', label='Direção Real (psi)')
    axs[3].axhline(y=np.rad2deg(target_psi), color='r', linestyle='--', label=f'Alvo psi ({np.rad2deg(target_psi):.1f}°)')
    axs[3].set_ylabel('Direção/Yaw (Graus)')
    axs[3].set_xlabel('Tempo (s)')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Salvar o gráfico
    file_name = f'grafico_resposta_{file_tag}.png'
    plt.savefig(file_name)
    print(f'Gráficos de resposta salvos como: {os.path.abspath(file_name)}')
    plt.close(fig)


# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL (MODIFICADA PARA O CENÁRIO DE VELOCIDADE)
# ==============================================================================
if __name__ == '__main__':
    os4 = Quadrotor()
    
    print("--- Simulador de Controle de Velocidade de Quadricóptero ---")
    
    # --- NOVO: Definição do "Vetor de Ataque" ---
    # Estes são os inputs do usuário
    ATTACK_SPEED_MPS = 3.0       # Velocidade desejada em metros/segundo
    ATTACK_ANGLE_DEG = 45.0      # Direção (0=X, 90=Y) em graus
    ATTACK_DURATION_S = 8.0      # Duração do movimento em segundos
    TARGET_ALTITUDE_M = 1.5      # Altitude a ser mantida
    
    # --- Conversão dos inputs do usuário em setpoints de controle ---
    
    # 1. Converter o ângulo de ataque em radianos
    attack_angle_rad = np.deg2rad(ATTACK_ANGLE_DEG)
    
    # 2. Decompor a velocidade em componentes do sistema de coordenadas global (vx, vy)
    target_vx = ATTACK_SPEED_MPS * np.cos(attack_angle_rad)
    target_vy = ATTACK_SPEED_MPS * np.sin(attack_angle_rad)
    
    # 3. Empacotar todos os alvos em um dicionário
    targets = {
        'vx': target_vx,
        'vy': target_vy,
        'z': TARGET_ALTITUDE_M,
        'psi': attack_angle_rad  # O drone deve "olhar" para onde está indo
    }
    
    print(f"Alvo de Ataque: {ATTACK_SPEED_MPS} m/s @ {ATTACK_ANGLE_DEG} graus por {ATTACK_DURATION_S}s")
    print(f"Setpoints Calculados: vx={target_vx:.2f}, vy={target_vy:.2f}, z={TARGET_ALTITUDE_M}, psi={np.rad2deg(attack_angle_rad):.1f}°")

    # --- Ganhos (Heurísticos) ---
    # Ganhos de Posição (Modo Antigo)
    pos_xy_gains = {'Kp': 0.8, 'Ki': 0.2, 'Kd': 0.5, 'anti_windup_limit': np.deg2rad(15)}
    # NOVOS Ganhos de Velocidade
    # Kp: 0.5 -> 1 m/s de erro de velocidade causa 0.5 rad (aprox 30 deg) de inclinação
    vel_xy_gains = {'Kp': 0.5, 'Ki': 0.1, 'Kd': 0.2, 'anti_windup_limit': np.deg2rad(25)}
    
    gain_dicts = {
        'z': {'Kp': 30.0, 'Ki': 25.0, 'Kd': 20.0, 'anti_windup_limit': 15.0},
        'attitude': {'Kp': 0.8, 'Ki': 0.5, 'Kd': 0.2, 'anti_windup_limit': 0.5},
        'yaw': {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.3, 'anti_windup_limit': 0.5},
        'position_xy': pos_xy_gains,
        'velocity_xy': vel_xy_gains # Novos ganhos para o novo modo
    }

    # --- Executa a simulação final com os ganhos otimizados ---
    print(f"\nExecutando simulação de {ATTACK_DURATION_S}s no modo VELOCIDADE...")
    sol = run_pid_simulation(os4, 
                             targets, 
                             ATTACK_DURATION_S, 
                             gain_dicts, 
                             control_mode='velocity', # <-- O PONTO CHAVE
                             dt=0.01, 
                             verbose=True)
    
    file_tag = f"attack_{ATTACK_SPEED_MPS}ms_{ATTACK_ANGLE_DEG}deg"
    
    # --- NOVO: Plotar os gráficos de resposta de VELOCIDADE ---
    plot_velocity_responses(sol, targets, file_tag)
    
    # --- Anima o resultado ---
    
    # Para a animação, calcular o "ponto de destino" projetado
    final_x_projetado = target_vx * ATTACK_DURATION_S
    final_y_projetado = target_vy * ATTACK_DURATION_S
    plot_target_marker = [final_x_projetado, final_y_projetado, TARGET_ALTITUDE_M]
    
    animate_and_save_3d(sol, file_tag, ATTACK_DURATION_S, os4, plot_target_marker)

    print("\nSimulação de vetor de ataque concluída e salva como GIF e PNG.")
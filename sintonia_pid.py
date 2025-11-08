import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import time
from scipy.optimize import minimize

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
        derivative = (error - self.prev_error) / dt
        D_term = self.Kd * derivative
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
# 3. FUNÇÃO DE SIMULAÇÃO PID (MODIFICADA)
# ==============================================================================
def run_pid_simulation(quad, target_pos, duration, gain_dicts, dt=0.01, verbose=False):
    """
    Executa a simulação de controle PID em loop fechado.
    AGORA RECEBE 'gain_dicts' como argumento.
    """
    
    # --- Ponto 1: Ganhos PID (Vindos do argumento) ---
    z_gains = gain_dicts['z']
    xy_gains = gain_dicts['xy']
    attitude_gains = gain_dicts['attitude']
    yaw_gains = gain_dicts['yaw']

    # --- Ponto 2: Inicialização dos Controladores ---
    pid_z = PIDController(setpoint=target_pos[2], **z_gains)
    pid_x = PIDController(setpoint=target_pos[0], **xy_gains)
    pid_y = PIDController(setpoint=target_pos[1], **xy_gains)
    pid_phi = PIDController(**attitude_gains)
    pid_theta = PIDController(**attitude_gains)
    pid_psi = PIDController(setpoint=target_pos[3], **yaw_gains)

    # --- Ponto 3: Misturador de Controle ---
    b, d, l = quad.b, quad.d, quad.l
    alloc_matrix = np.array([[b, b, b, b],
                             [0, -l*b, 0, l*b],
                             [l*b, 0, -l*b, 0],
                             [d, -d, d, -d]])
    inv_alloc_matrix = np.linalg.inv(alloc_matrix)
    max_thrust_per_motor = (quad.m * quad.g * 2) / 4.0
    max_w_sq = max_thrust_per_motor / quad.b

    # --- Ponto 4: O Loop de Simulação ---
    if verbose:
        print(f"Iniciando simulação controlada para: {target_pos[:3]}...")
    
    state = quad.initial_state.copy()
    time = 0.0
    times = [time]
    states = [state]
    num_steps = int(duration / dt)
    
    for i in range(num_steps):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        # 1. Loop Externo
        thrust_correction = pid_z.update(z, dt)
        total_thrust = (quad.m * quad.g) + thrust_correction
        theta_des = pid_x.update(x, dt)
        phi_des = -pid_y.update(y, dt)
        psi_des = target_pos[3]
        
        max_angle = np.deg2rad(30)
        phi_des = np.clip(phi_des, -max_angle, max_angle)
        theta_des = np.clip(theta_des, -max_angle, max_angle)

        # 2. Loop Interno
        pid_phi.set_setpoint(phi_des)
        pid_theta.set_setpoint(theta_des)
        pid_psi.set_setpoint(psi_des)
        tau_phi = pid_phi.update(phi, dt)
        tau_theta = pid_theta.update(theta, dt)
        tau_psi = pid_psi.update(psi, dt)

        # 3. Misturador
        control_vector = np.array([total_thrust, tau_phi, tau_theta, tau_psi])
        w_squared = inv_alloc_matrix @ control_vector
        w_squared = np.clip(w_squared, 0, max_w_sq)
        inputs = np.sqrt(w_squared)
        
        # 4. Atualização da Dinâmica
        state_dot = quad.dynamics(time, state, inputs)
        state = state + np.array(state_dot) * dt
        
        # 5. Armazenamento
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
# 4. NOVAS FUNÇÕES: HELPERS DE OTIMIZAÇÃO
# ==============================================================================

def unpack_gains(gains_vector):
    """Converte um vetor de 12 elementos em um dicionário de ganhos."""
    # Garante que nenhum ganho seja negativo
    gains_vector = np.maximum(gains_vector, 0) 
    
    z_gains = {'Kp': gains_vector[0], 'Ki': gains_vector[1], 'Kd': gains_vector[2], 'anti_windup_limit': 15.0}
    xy_gains = {'Kp': gains_vector[3], 'Ki': gains_vector[4], 'Kd': gains_vector[5], 'anti_windup_limit': np.deg2rad(15)}
    attitude_gains = {'Kp': gains_vector[6], 'Ki': gains_vector[7], 'Kd': gains_vector[8], 'anti_windup_limit': 0.5}
    yaw_gains = {'Kp': gains_vector[9], 'Ki': gains_vector[10], 'Kd': gains_vector[11], 'anti_windup_limit': 0.5}
    
    return {'z': z_gains, 'xy': xy_gains, 'attitude': attitude_gains, 'yaw': yaw_gains}

def pack_gains(gain_dicts):
    """Converte um dicionário de ganhos em um vetor de 12 elementos."""
    g = gain_dicts
    return np.array([
        g['z']['Kp'], g['z']['Ki'], g['z']['Kd'],
        g['xy']['Kp'], g['xy']['Ki'], g['xy']['Kd'],
        g['attitude']['Kp'], g['attitude']['Ki'], g['attitude']['Kd'],
        g['yaw']['Kp'], g['yaw']['Ki'], g['yaw']['Kd']
    ])

def cost_function(gains_vector, quad, target_pos, duration, dt):
    """
    A função de custo que o otimizador tentará minimizar.
    Roda uma simulação e retorna um único número (custo).
    """
    
    # 1. Converter o vetor de ganhos em dicionários
    gain_dicts = unpack_gains(gains_vector)
    
    # 2. Rodar a simulação
    try:
        sol = run_pid_simulation(quad, target_pos, duration, gain_dicts, dt, verbose=False)
    except Exception as e:
        # Se a simulação falhar (ex: ganhos instáveis causam 'nan')
        # retorne um custo infinitamente alto.
        # print(f"Simulação falhou com ganhos: {gains_vector}. Custo: Infinito.")
        return np.inf

    # 3. Analisar os resultados e calcular o custo
    t = sol.t
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    target_x, target_y, target_z = target_pos[0], target_pos[1], target_pos[2]

    # --- Calcular Erros ---
    error_x = target_x - x
    error_y = target_y - y
    error_z = target_z - z

    # --- Custo 1: ITAE (Integral of Time-weighted Absolute Error) ---
    # Penaliza erros que persistem
    itae_x = np.trapezoid(t * np.abs(error_x), t)
    itae_y = np.trapezoid(t * np.abs(error_y), t)
    itae_z = np.trapezoid(t * np.abs(error_z), t)
    
    # --- Custo 2: Penalidade de Sobressinal (Overshoot) ---
    # Penalidade enorme se o overshoot for > 10%
    max_x = np.max(x)
    max_y = np.max(y)
    max_z = np.max(z)
    
    # Overshoot relativo (ex: 0.15 para 15%)
    # Lida com o caso de alvo ser 0 ou negativo (embora não para z)
    overshoot_x = (max_x - target_x) / target_x if target_x > 1e-3 else 0
    overshoot_y = (max_y - target_y) / target_y if target_y > 1e-3 else 0
    overshoot_z = (max_z - target_z) / target_z if target_z > 1e-3 else 0

    # Penalidade é zero se overshoot < 0.10 (10%)
    # E cresce quadraticamente se for maior
    penalty_x = max(0, (overshoot_x - 0.10))**2 * 10000
    penalty_y = max(0, (overshoot_y - 0.10))**2 * 10000
    penalty_z = max(0, (overshoot_z - 0.10))**2 * 10000
    
    # --- Custo 3: Penalidade de Erro de Estado Estacionário ---
    # Penaliza se não chegar ao alvo no final
    ss_error_x = np.abs(error_x[-1]) * 100
    ss_error_y = np.abs(error_y[-1]) * 100
    ss_error_z = np.abs(error_z[-1]) * 100

    # --- Custo Total ---
    total_cost = (
        (itae_x + itae_y + itae_z) +
        (penalty_x + penalty_y + penalty_z) +
        (ss_error_x + ss_error_y + ss_error_z)
    )

    # Otimizadores podem lidar mal com 'nan' ou 'inf'
    if not np.isfinite(total_cost):
        return np.inf
        
    return total_cost


# ==============================================================================
# 5. FUNÇÃO DE ANIMAÇÃO (Sem mudanças)
# ==============================================================================
def animate_and_save_3d(sol, movement_type, duration, quad_params, target_pos):
    """
    Cria e salva uma animação 3D da trajetória do quadricóptero como um GIF.
    """
    x_data, y_data, z_data = sol.y[0], sol.y[1], sol.y[2]
    phi_data, theta_data, psi_data = sol.y[6], sol.y[7], sol.y[8]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    max_range = max(np.max(x_data) - np.min(x_data), 
                    np.max(y_data) - np.min(y_data), 
                    np.max(z_data) - np.min(z_data),
                    1.0, # Garante zoom mínimo
                    target_pos[0], target_pos[1], target_pos[2]) # Garante que o alvo esteja visível
    
    mid_x = (np.max(x_data) + np.min(x_data)) / 2
    mid_y = (np.max(y_data) + np.min(y_data)) / 2
    mid_z = (np.max(z_data) + np.min(z_data)) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    if (mid_z - max_range) > 0:
        ax.set_zlim(0, mid_z + max_range)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Animação 3D: {movement_type.replace("_", " ").title()}', fontsize=16)

    X_ground, Y_ground = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10), 
                                     np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10))
    Z_ground = np.zeros_like(X_ground)
    ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='gray')

    trail, = ax.plot([], [], [], '-', lw=2, color='blue', label='Rastro da Trajetória')
    line1, = ax.plot([], [], [], '-', lw=3, color='red') 
    line2, = ax.plot([], [], [], '-', lw=3, color='red', label='Quadricóptero (X)')
    
    # Adiciona o ponto de alvo
    ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'go', markersize=10, label='Alvo')
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
    
    file_name = f'animacao_3d_{movement_type}_controlado.gif'
    print(f'Criando animação: {file_name}...')
    ani.save(file_name, writer='pillow', fps=fps)
    print(f'Animação salva como: {os.path.abspath(file_name)}')
    plt.close(fig)

# ==============================================================================
# 6. NOVA FUNÇÃO: PLOTAGEM DAS RESPOSTAS
# ==============================================================================
def plot_pid_responses(sol, target_pos, file_tag):
    """
    Cria e salva gráficos de linha para as respostas dos eixos X, Y e Z.
    """
    print(f"Gerando gráficos de resposta para: {file_tag}...")
    
    t = sol.t
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    target_x, target_y, target_z = target_pos[0], target_pos[1], target_pos[2]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Respostas do Controlador PID - {file_tag.replace("_", " ").title()}', fontsize=16)

    # --- Eixo X ---
    axs[0].plot(t, x, 'b-', label='Posição X Real')
    axs[0].axhline(y=target_x, color='r', linestyle='--', label='Alvo X')
    # Linha de 10% de overshoot
    overshoot_line_x = target_x * 1.1
    axs[0].axhline(y=overshoot_line_x, color='g', linestyle=':', label='Limite Overshoot (10%)')
    axs[0].set_ylabel('Posição X (m)')
    axs[0].legend()
    axs[0].grid(True)

    # --- Eixo Y ---
    axs[1].plot(t, y, 'b-', label='Posição Y Real')
    axs[1].axhline(y=target_y, color='r', linestyle='--', label='Alvo Y')
    # Linha de 10% de overshoot
    overshoot_line_y = target_y * 1.1
    axs[1].axhline(y=overshoot_line_y, color='g', linestyle=':', label='Limite Overshoot (10%)')
    axs[1].set_ylabel('Posição Y (m)')
    axs[1].legend()
    axs[1].grid(True)

    # --- Eixo Z ---
    axs[2].plot(t, z, 'b-', label='Posição Z Real (Altitude)')
    axs[2].axhline(y=target_z, color='r', linestyle='--', label='Alvo Z')
    # Linha de 10% de overshoot
    overshoot_line_z = target_z * 1.1
    axs[2].axhline(y=overshoot_line_z, color='g', linestyle=':', label='Limite Overshoot (10%)')
    axs[2].set_ylabel('Posição Z (m)')
    axs[2].set_xlabel('Tempo (s)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Salvar o gráfico
    file_name = f'grafico_resposta_{file_tag}.png'
    plt.savefig(file_name)
    print(f'Gráficos de resposta salvos como: {os.path.abspath(file_name)}')
    
    # Opcional: mostrar o gráfico
    # plt.show()
    plt.close(fig)


# ==============================================================================
# 7. EXECUÇÃO PRINCIPAL (MODIFICADA PARA SINTONIA E PLOTAGEM)
# ==============================================================================
if __name__ == '__main__':
    os4 = Quadrotor()
    
    print("--- Simulador de Controle PID de Quadricóptero com Auto-Tuning ---")
    
    # --- Definição do Alvo ---
    # Otimizaremos os ganhos para este alvo específico
    target_x, target_y, target_z = 2.0, 2.0, 1.0
    target_psi = 0.0
    target_pos = [target_x, target_y, target_z, target_psi]
    duration = 10.0
    dt = 0.01

    # --- Ganhos Iniciais (Seu palpite heurístico) ---
    # É importante começar de um ponto razoável
    initial_gain_dicts = {
        'z': {'Kp': 30.0, 'Ki': 25.0, 'Kd': 20.0, 'anti_windup_limit': 15.0},
        'xy': {'Kp': 0.8, 'Ki': 0.2, 'Kd': 0.5, 'anti_windup_limit': np.deg2rad(15)},
        'attitude': {'Kp': 0.8, 'Ki': 0.5, 'Kd': 0.2, 'anti_windup_limit': 0.5},
        'yaw': {'Kp': 0.1, 'Ki': 0.01, 'Kd': 0.05, 'anti_windup_limit': 0.5}
    }
    initial_gains_vector = pack_gains(initial_gain_dicts)

    # --- Execução da Otimização ---
    print(f"Iniciando sintonia automática para o alvo: {target_pos[:3]}...")
    print("Isso pode levar alguns minutos, pois várias simulações serão executadas.")
    
    start_time = time.time()
    
    # Argumentos extras para passar para a 'cost_function'
    optimization_args = (os4, target_pos, duration, dt)

    # Chamada do otimizador Nelder-Mead
    # 'maxiter' é o número de simulações. 100-200 é um bom começo.
    result = minimize(cost_function, 
                      initial_gains_vector, 
                      args=optimization_args, 
                      method='Nelder-Mead',
                      options={'maxiter': 100, 'adaptive': True, 'disp': True})
    
    end_time = time.time()
    print(f"Sintonia concluída em {end_time - start_time:.2f} segundos.")

    # --- Resultados da Otimização ---
    if not result.success:
        print("\nAVISO: O otimizador pode não ter convergido com sucesso.")
        print(f"Mensagem: {result.message}")
    
    tuned_gains_vector = np.maximum(result.x, 0) # Garante que não há ganhos negativos
    tuned_gain_dicts = unpack_gains(tuned_gains_vector)

    print("\n--- Ganhos Originais (Heurísticos) ---")
    print(initial_gains_vector)
    
    print("\n--- Ganhos Otimizados (Sintonizados) ---")
    print(tuned_gains_vector)
    print("\n(Kp, Ki, Kd)")
    print(f"Z:        {tuned_gain_dicts['z']['Kp']:.3f}, {tuned_gain_dicts['z']['Ki']:.3f}, {tuned_gain_dicts['z']['Kd']:.3f}")
    print(f"XY:       {tuned_gain_dicts['xy']['Kp']:.3f}, {tuned_gain_dicts['xy']['Ki']:.3f}, {tuned_gain_dicts['xy']['Kd']:.3f}")
    print(f"Atitude:  {tuned_gain_dicts['attitude']['Kp']:.3f}, {tuned_gain_dicts['attitude']['Ki']:.3f}, {tuned_gain_dicts['attitude']['Kd']:.3f}")
    print(f"Yaw:      {tuned_gain_dicts['yaw']['Kp']:.3f}, {tuned_gain_dicts['yaw']['Ki']:.3f}, {tuned_gain_dicts['yaw']['Kd']:.3f}")


    # --- Executa a simulação final com os ganhos otimizados ---
    print("\nExecutando simulação final com ganhos otimizados...")
    sol = run_pid_simulation(os4, target_pos, duration, tuned_gain_dicts, dt, verbose=True)
    
    file_tag = f"goto_{target_x}x_{target_y}y_{target_z}z_TUNED"
    
    # --- NOVO: Plotar os gráficos de resposta ---
    plot_pid_responses(sol, target_pos, file_tag)
    
    # Anima o resultado
    animate_and_save_3d(sol, file_tag, duration, os4, target_pos)

    print("\nSimulação otimizada concluída e salva como GIF e PNG.")
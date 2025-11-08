import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# ==============================================================================
# 1. CLASSE DO CONTROLADOR PID
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
        
        # Limite 'anti-windup' para evitar que o termo integral cresça indefinidamente
        self.anti_windup_limit = anti_windup_limit

    def set_setpoint(self, setpoint):
        """Define o valor desejado (alvo)."""
        self.setpoint = setpoint

    def update(self, current_value, dt):
        """Calcula a saída do controlador com base no valor atual."""
        error = self.setpoint - current_value
        
        # Termo Proporcional
        P_term = self.Kp * error
        
        # Termo Integral (com anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.anti_windup_limit, self.anti_windup_limit)
        I_term = self.Ki * self.integral
        
        # Termo Derivativo (com filtro para evitar picos)
        # Usamos (erro_atual - erro_anterior) / dt
        derivative = (error - self.prev_error) / dt
        D_term = self.Kd * derivative
        
        # Atualiza o erro anterior
        self.prev_error = error
        
        # Saída total
        output = P_term + I_term + D_term
        return output

# ==============================================================================
# 2. CLASSE DO QUADRICÓPTERO (MODIFICADA)
# ==============================================================================
class Quadrotor:
    """
    Classe que representa o modelo dinâmico de um quadricóptero.
    MODIFICADA: A função 'dynamics' agora aceita 'inputs' (w1, w2, w3, w4).
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
        
        # Modelo de contato com o solo
        self.ground_k = 500.0
        self.ground_c = 100.0

        # Estado inicial (sempre começa do chão)
        self.initial_state = np.zeros(12)

    def dynamics(self, t, state, inputs):
        """
        Define as equações diferenciais.
        'inputs' é um array [w1, w2, w3, w4] com as velocidades dos rotores.
        """
        w1, w2, w3, w4 = inputs
        
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        # Forças e Torques (calculados a partir das entradas w_i)
        T = self.b * (w1**2 + w2**2 + w3**2 + w4**2)
        tau_phi = self.l * self.b * (w4**2 - w2**2)
        tau_theta = self.l * self.b * (w1**2 - w3**2)
        tau_psi = self.d * (w1**2 - w2**2 + w3**2 - w4**2)
        
        # Matriz de Rotação (Corpo -> Inercial)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
                        
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
                        
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

        # Dinâmica rotacional (equações de Euler)
        p_dot = (tau_phi - q * r * (self.Izz - self.Iyy)) / self.Ixx
        q_dot = (tau_theta - p * r * (self.Ixx - self.Izz)) / self.Iyy
        r_dot = (tau_psi - p * q * (self.Iyy - self.Ixx)) / self.Izz
        
        # Conversão de taxas do corpo (p,q,r) para taxas de Euler (phi_dot, ...)
        T_euler = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        
        if abs(np.cos(theta)) < 1e-6:
            # Evita singularidade (gimbal lock)
            euler_rates = np.zeros(3)
        else:
            euler_rates = T_euler @ np.array([p, q, r])

        phi_dot = euler_rates[0]
        theta_dot = euler_rates[1]
        psi_dot = euler_rates[2]

        return [vx, vy, vz, ax, ay, az, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]

# ==============================================================================
# 3. FUNÇÃO DE SIMULAÇÃO PID (O NOVO NÚCLEO)
# ==============================================================================
def run_pid_simulation(quad, target_pos, duration, dt=0.01):
    """
    Executa a simulação de controle PID em loop fechado.
    'target_pos' é [x_des, y_des, z_des, psi_des].
    """
    
    # --- Ponto 1: Ganhos PID (A "Sintonia Heurística") ---
    # Estes são os ganhos "projetados heuristicamente" (na prática, ajustados
    # por tentativa e erro). Eles são o coração do seu pedido.
    
    # Loop Externo (Posição)
    # Z (Altitude)
    z_gains = {'Kp': 46.833, 'Ki': 16.85, 'Kd': 15.101, 'anti_windup_limit': 15.0}
    # X/Y (Posição Horizontal) -> Saída é um ÂNGULO desejado
    xy_gains = {'Kp': 0.845, 'Ki': 0.195, 'Kd': 0.485, 'anti_windup_limit': np.deg2rad(15)} # Limite de inclinação

    # Loop Interno (Atitude)
    # Roll/Pitch -> Saída é um TORQUE
    attitude_gains = {'Kp': 0.738, 'Ki': 0.492, 'Kd': 0.207, 'anti_windup_limit': 0.5}
    # Yaw -> Saída é um TORQUE
    yaw_gains = {'Kp': 0.109, 'Ki': 0.011, 'Kd': 0.051, 'anti_windup_limit': 0.5}

    # --- Ponto 2: Inicialização dos Controladores ---
    
    # Controladores de Posição (Loop Externo)
    pid_z = PIDController(setpoint=target_pos[2], **z_gains)
    pid_x = PIDController(setpoint=target_pos[0], **xy_gains)
    pid_y = PIDController(setpoint=target_pos[1], **xy_gains)

    # Controladores de Atitude (Loop Interno)
    pid_phi = PIDController(**attitude_gains)   # Setpoint é dinâmico (vem do PID de Y)
    pid_theta = PIDController(**attitude_gains) # Setpoint é dinâmico (vem do PID de X)
    pid_psi = PIDController(setpoint=target_pos[3], **yaw_gains)

    # --- Ponto 3: Misturador de Controle (Inverso da Dinâmica) ---
    # Mapeia [T, tau_phi, tau_theta, tau_psi] -> [w1^2, w2^2, w3^2, w4^2]
    b, d, l = quad.b, quad.d, quad.l
    # Matriz de alocação: [T; tau_phi; tau_theta; tau_psi] = A * [w1^2; ...; w4^2]
    alloc_matrix = np.array([[b, b, b, b],
                             [0, -l*b, 0, l*b],
                             [l*b, 0, -l*b, 0],
                             [d, -d, d, -d]])
    # Usamos a inversa para encontrar as velocidades dos motores
    inv_alloc_matrix = np.linalg.inv(alloc_matrix)

    # Velocidade máxima do motor (para clamp/clip)
    # Assumindo que o empuxo máximo é ~2x o peso
    max_thrust_per_motor = (quad.m * quad.g * 2) / 4.0
    max_w_sq = max_thrust_per_motor / quad.b
    max_w = np.sqrt(max_w_sq)

    # --- Ponto 4: O Loop de Simulação ---
    
    print(f"Iniciando simulação controlada para: {target_pos[:3]}...")
    
    # Inicialização do estado
    state = quad.initial_state.copy()
    time = 0.0
    
    # Armazenamento para resultados
    times = [time]
    states = [state]
    
    num_steps = int(duration / dt)
    
    for i in range(num_steps):
        # Estado atual
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        # --- Loop de Controle ---
        
        # 1. Loop Externo (Posição) -> Calcula Setpoints para o Loop Interno
        
        # Empuxo Total (calculado pelo PID de altitude)
        # Adicionamos 'feed-forward' (m*g) para compensar a gravidade
        thrust_correction = pid_z.update(z, dt)
        total_thrust = (quad.m * quad.g) + thrust_correction
        
        # Ângulos de Roll/Pitch desejados (calculados pelos PIDs X e Y)
        # Nota: Um erro em X positivo requer um Pitch (theta) positivo.
        #       Um erro em Y positivo requer um Roll (phi) NEGATIVO.
        theta_des = pid_x.update(x, dt)
        phi_des = -pid_y.update(y, dt)
        psi_des = target_pos[3] # Fixo
        
        # Limita os ângulos (margem de erro/segurança)
        max_angle = np.deg2rad(30) # Limite de 30 graus
        phi_des = np.clip(phi_des, -max_angle, max_angle)
        theta_des = np.clip(theta_des, -max_angle, max_angle)

        # 2. Loop Interno (Atitude) -> Calcula Torques desejados
        
        # Define os setpoints dinâmicos
        pid_phi.set_setpoint(phi_des)
        pid_theta.set_setpoint(theta_des)
        pid_psi.set_setpoint(psi_des)
        
        # Calcula os torques
        tau_phi = pid_phi.update(phi, dt)
        tau_theta = pid_theta.update(theta, dt)
        tau_psi = pid_psi.update(psi, dt)

        # 3. Misturador de Controle -> Calcula velocidades dos motores
        
        # Vetor de controle desejado
        control_vector = np.array([total_thrust, tau_phi, tau_theta, tau_psi])
        
        # Calcula o quadrado das velocidades angulares
        w_squared = inv_alloc_matrix @ control_vector
        
        # Garante que as velocidades sejam positivas e dentro dos limites
        w_squared = np.clip(w_squared, 0, max_w_sq)
        
        # Entradas finais para o modelo dinâmico
        inputs = np.sqrt(w_squared)
        # inputs = np.clip(inputs, 0, max_w) # O clip em w_squared já resolve
        
        # --- Fim do Loop de Controle ---
        
        # 4. Atualização da Dinâmica (Integração de Euler)
        # (Usar RK4 seria mais preciso, mas Euler é mais simples)
        state_dot = quad.dynamics(time, state, inputs)
        state = state + np.array(state_dot) * dt
        
        # 5. Armazenamento
        time += dt
        times.append(time)
        states.append(state.copy())

    print("Simulação concluída.")
    
    # Formata a saída para ser compatível com a função de animação
    # (solve_ivp retorna (N_estados, N_pontos), então transpomos)
    solution = type('Solution', (object,), {})() # Objeto 'dummy'
    solution.t = np.array(times)
    solution.y = np.array(states).T # Transpor
    
    return solution

# ==============================================================================
# 4. FUNÇÃO DE ANIMAÇÃO (COMO NO SEU CÓDIGO ORIGINAL)
# ==============================================================================
def animate_and_save_3d(sol, movement_type, duration, quad_params):
    """
    Cria e salva uma animação 3D da trajetória do quadricóptero como um GIF.
    O quadricóptero é desenhado como um "X".
    """
    x_data, y_data, z_data = sol.y[0], sol.y[1], sol.y[2]
    phi_data, theta_data, psi_data = sol.y[6], sol.y[7], sol.y[8]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Encontra os limites máximos dos dados para centralizar os eixos
    max_range = max(np.max(x_data) - np.min(x_data), 
                    np.max(y_data) - np.min(y_data), 
                    np.max(z_data) - np.min(z_data))
    if max_range < 1.0: max_range = 1.0 # Garante um zoom mínimo
    
    mid_x = (np.max(x_data) + np.min(x_data)) / 2
    mid_y = (np.max(y_data) + np.min(y_data)) / 2
    mid_z = (np.max(z_data) + np.min(z_data)) / 2
    
    # Define limites dinâmicos, mas centrados e simétricos
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # Garante que o chão (z=0) esteja visível
    if (mid_z - max_range) > 0:
        ax.set_zlim(0, mid_z + max_range)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Animação 3D: {movement_type.replace("_", " ").title()}', fontsize=16)

    # Adiciona um plano para representar o chão
    X_ground, Y_ground = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10), 
                                     np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 10))
    Z_ground = np.zeros_like(X_ground)
    ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='gray')

    trail, = ax.plot([], [], [], '-', lw=2, color='blue', label='Rastro da Trajetória')
    line1, = ax.plot([], [], [], '-', lw=3, color='red') 
    line2, = ax.plot([], [], [], '-', lw=3, color='red', label='Quadricóptero (X)')
    
    # Adiciona um ponto de alvo
    target_point = sol.y[:3, -1] # Pega o último ponto como referência, mas usamos o setpoint
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
        # Pula frames para acelerar a animação se for muito longa
        idx = frame * 5 # Pula 5 frames
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

    num_frames_anim = len(sol.t) // 5 # Ajustado por causa do pulo de frames
    fps = int(num_frames_anim / duration)
    if fps < 15: fps = 15
    if fps > 60: fps = 60

    ani = FuncAnimation(fig, update, frames=num_frames_anim, init_func=init, blit=True)
    
    file_name = f'animacao_3d_{movement_type}_controlado.gif'
    print(f'Criando animação: {file_name}...')
    ani.save(file_name, writer='pillow', fps=fps)
    print(f'Animação salva como: {os.path.abspath(file_name)}')
    plt.close(fig)

# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    os4 = Quadrotor()
    
    print("--- Simulador de Controle PID de Quadricóptero ---")
    print("O quadricóptero começará no chão (0, 0, 0).")
    
    try:
        target_x = float(input("Digite a coordenada X desejada (ex: 2.0): "))
        target_y = float(input("Digite a coordenada Y desejada (ex: 3.0): "))
        target_z = float(input("Digite a coordenada Z desejada (ex: 1.5): "))
    except ValueError:
        print("Entrada inválida. Usando valores padrão (2, 2, 1).")
        target_x, target_y, target_z = 2.0, 2.0, 1.0

    target_psi = 0.0 # Manter a guinada (yaw) em 0 graus
    
    target_pos = [target_x, target_y, target_z, target_psi]
    
    # Duração da simulação
    duration = 10.0 # Damos mais tempo para o controlador estabilizar
    
    # Executa a simulação de controle
    sol = run_pid_simulation(os4, target_pos, duration)
    
    # Anima o resultado
    file_tag = f"goto_{target_x}x_{target_y}y_{target_z}z"
    animate_and_save_3d(sol, file_tag, duration, os4)

    print("\nSimulação concluída e salva como GIF.")
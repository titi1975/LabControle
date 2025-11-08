import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import time

# Importações de Álgebra Linear para LQR
from scipy.linalg import solve_continuous_are, inv

# ==============================================================================
# 1. CLASSE DO QUADRICÓPTERO (Sem mudanças)
# ==============================================================================
# Esta classe representa o "mundo real" não-linear.
# Não a modificamos.
class Quadrotor:
# ... (código existente da classe Quadrotor sem mudanças) ...
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
        'inputs' é um array [w1, w2, w3, w4] com as velocidades dos rotores.
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
# 2. CLASSE ATUALIZADA: CONTROLADOR LQI (LQR com Ação Integral)
# ==============================================================================
class LQIController:
    """
    Calcula a matriz de ganho K do LQI (LQR + Integral)
    para eliminar o erro de estado estacionário.
    """
    def __init__(self, quad):
        # Armazena os parâmetros físicos
        self.m = quad.m
        self.g = quad.g
        self.Ixx = quad.Ixx
        self.Iyy = quad.Iyy
        self.Izz = quad.Izz
        
        # Constrói as matrizes A e B do modelo linearizado ORIGINAL
        self._build_linear_model()
        
        # K será dividido em K_p (proporcional) e K_i (integral)
        self.K_p = None
        self.K_i = None 

    def _build_linear_model(self):
        """
        Constrói as matrizes A (12x12) e B (12x4) baseadas na derivação.
        """
        
        # Matriz A (12x12)
        self.A = np.zeros((12, 12))
        self.A[0, 3] = 1.0; self.A[1, 4] = 1.0; self.A[2, 5] = 1.0
        self.A[3, 7] = self.g; self.A[4, 6] = -self.g
        self.A[6, 9] = 1.0; self.A[7, 10] = 1.0; self.A[8, 11] = 1.0
        
        # Matriz B (12x4)
        self.B = np.zeros((12, 4))
        self.B[5, 0] = 1.0 / self.m
        self.B[9, 1] = 1.0 / self.Ixx
        self.B[10, 2] = 1.0 / self.Iyy
        self.B[11, 3] = 1.0 / self.Izz

    def build_augmented_system(self):
        """
        Cria as matrizes aumentadas A_aug (15x15) e B_aug (15x4)
        para incluir os 3 estados integrais (ix, iy, iz).
        """
        
        # Dimensões
        n_states = self.A.shape[0] # 12
        n_integrators = 3 # ix, iy, iz
        n_inputs = self.B.shape[1] # 4
        
        # Novas matrizes aumentadas (15x15 e 15x4)
        self.A_aug = np.zeros((n_states + n_integrators, n_states + n_integrators))
        self.B_aug = np.zeros((n_states + n_integrators, n_inputs))
        
        # Bloco superior esquerdo: A original (12x12)
        self.A_aug[:n_states, :n_states] = self.A
        
        # Bloco inferior esquerdo: C_int (3x12)
        # Define a dinâmica do integrador: ix_dot = x, iy_dot = y, iz_dot = z
        self.A_aug[n_states:, 0] = 1.0 # ix_dot = x
        self.A_aug[n_states+1, 1] = 1.0 # iy_dot = y
        self.A_aug[n_states+2, 2] = 1.0 # iz_dot = z
        
        # Bloco superior direito: B original (12x4)
        self.B_aug[:n_states, :] = self.B
        
        # Outros blocos são zeros (já inicializados)
        
        print("Matrizes aumentadas A_aug (15x15) e B_aug (15x4) construídas.")

    def tune(self, Q_aug, R_aug):
        """
        Calcula a matriz de ganho K_aug (4x15) ótima do LQI.
        Q_aug (15x15) = Matriz de custo de estado aumentado
        R_aug (4x4)   = Matriz de custo de entrada
        """
        
        # Construir as matrizes aumentadas primeiro
        self.build_augmented_system()
        
        print("Sintonizando LQI (Resolvendo equação de Riccati aumentada)...")
        
        # Resolve a ARE contínua para o sistema aumentado
        P_aug = solve_continuous_are(self.A_aug, self.B_aug, Q_aug, R_aug)
        
        # Calcula a matriz de ganho K aumentada (4x15)
        K_aug = inv(R_aug) @ self.B_aug.T @ P_aug
        
        # Divide K_aug em parte Proporcional (K_p) e Integral (K_i)
        n_states = self.A.shape[0] # 12
        self.K_p = K_aug[:, :n_states]  # Ganhos (4x12) para os estados originais
        self.K_i = K_aug[:, n_states:]  # Ganhos (4x3) para os estados integrais
        
        print("Matrizes de ganho LQI K_p (4x12) e K_i (4x3) calculadas.")
        return self.K_p, self.K_i

# ==============================================================================
# 3. FUNÇÃO DE SIMULAÇÃO ATUALIZADA (Baseada no LQI)
# ==============================================================================
def run_lqi_simulation(quad, controller, target_pos, duration, dt=0.01):
    """
    Executa a simulação em loop fechado usando o controlador LQI.
    """
    
    if controller.K_p is None:
        print("Erro: O controlador LQI não foi sintonizado. Chame controller.tune(Q, R) primeiro.")
        return None

    # Ganhos LQI (proporcional e integral)
    K_p = controller.K_p
    K_i = controller.K_i
    
    # --- Ponto 1: Definir Alvos e Feed-forward ---
    
    # Vetor de estado alvo (equilíbrio)
    x_eq = np.zeros(12)
    x_eq[0] = target_pos[0] # x_alvo
    x_eq[1] = target_pos[1] # y_alvo
    x_eq[2] = target_pos[2] # z_alvo
    x_eq[8] = target_pos[3] # psi_alvo
    
    # Vetor de entrada de equilíbrio (Feed-forward)
    u_virtual_eq = np.array([quad.m * quad.g, 0, 0, 0])

    # --- Ponto 2: Misturador de Controle (Inalterado) ---
    b, d, l = quad.b, quad.d, quad.l
    alloc_matrix = np.array([[b, b, b, b],
                             [0, -l*b, 0, l*b],
                             [l*b, 0, -l*b, 0],
                             [d, -d, d, -d]])
    inv_alloc_matrix = np.linalg.inv(alloc_matrix)
    max_thrust_per_motor = (quad.m * quad.g * 2) / 4.0
    max_w_sq = max_thrust_per_motor / quad.b
    max_w = np.sqrt(max_w_sq)

    # --- Ponto 3: O Loop de Simulação ---
    print(f"Iniciando simulação LQI para: {target_pos[:3]}...")
    
    state = quad.initial_state.copy()
    
    # NOVO: Inicializar o estado do integrador
    integral_error = np.zeros(3) # [ix, iy, iz]
    
    time = 0.0
    times = [time]
    states = [state]
    num_steps = int(duration / dt)
    
    for i in range(num_steps):
        
        # --- Loop de Controle LQI ---
        
        # 1. Calcular o erro de estado (δx = x - x_eq)
        state_error = state - x_eq
        
        # 2. Atualizar o erro integral (com anti-windup simples)
        # Apenas integre se não estivermos saturados (simplificação)
        # Uma lógica anti-windup melhor seria necessária para robustez
        current_pos_error = state_error[[0, 1, 2]] # Erro em x, y, z
        integral_error += current_pos_error * dt
        
        # Limitar o integrador (anti-windup)
        integral_limit = 2.0
        integral_error = np.clip(integral_error, -integral_limit, integral_limit)

        # 3. Calcular o desvio de controle (δu = -K_p*δx - K_i*δi)
        control_p = -K_p @ state_error
        control_i = -K_i @ integral_error
        control_deviation = control_p + control_i
        
        # 4. Calcular a entrada virtual total (u = u_eq + δu)
        control_vector = u_virtual_eq + control_deviation
        
        # --- Fim do Loop de Controle ---

        # 5. Misturador -> Converter entradas virtuais em w_i
        w_squared = inv_alloc_matrix @ control_vector
        w_squared = np.clip(w_squared, 0, max_w_sq)
        inputs = np.sqrt(w_squared)
        
        # 6. Atualização da Dinâmica (Mundo Real Não-Linear)
        state_dot = quad.dynamics(time, state, inputs)
        state = state + np.array(state_dot) * dt
        
        # 7. Armazenamento
        time += dt
        times.append(time)
        states.append(state.copy())

    print("Simulação LQI concluída.")
    
    solution = type('Solution', (object,), {})()
    solution.t = np.array(times)
    solution.y = np.array(states).T
    
    return solution

# ==============================================================================
# 4. FUNÇÃO DE ANIMAÇÃO (Inalterada)
# ==============================================================================
def animate_and_save_3d(sol, movement_type, duration, quad_params, target_pos):
# ... (código existente da função animate_and_save_3d sem mudanças) ...
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
    
    ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], 'go', markersize=10, label='Alvo')
    ax.legend(loc='upper left')
    arm_length = quad_params.l 
    p_fl = np.array([arm_length, arm_length, 0]) / np.sqrt(2)
    p_fr = np.array([arm_length, -arm_length, 0]) / np.sqrt(2)
    p_rl = np.array([-arm_length, arm_length, 0]) / np.sqrt(2)
    # --- ALTERAÇÃO APLICADA (Correção de Bug) ---
    p_rr = np.array([-arm_length, -arm_length, 0]) / np.sqrt(2) # ERRO CORRIGIDO: Removido '=='

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
# 5. FUNÇÃO DE PLOTAGEM (Renomeada)
# ==============================================================================
def plot_responses(sol, target_pos, file_tag):
# ... (código existente da função plot_responses sem mudanças) ...
    """
    Cria e salva gráficos de linha para as respostas dos eixos X, Y e Z.
    """
    print(f"Gerando gráficos de resposta para: {file_tag}...")
    
    t = sol.t
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    target_x, target_y, target_z = target_pos[0], target_pos[1], target_pos[2]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Respostas do Controlador LQI - {file_tag.replace("_", " ").title()}', fontsize=16)

    # --- Eixo X ---
    axs[0].plot(t, x, 'b-', label='Posição X Real')
    axs[0].axhline(y=target_x, color='r', linestyle='--', label='Alvo X')
    axs[0].set_ylabel('Posição X (m)')
    axs[0].legend()
    axs[0].grid(True)

    # --- Eixo Y ---
    axs[1].plot(t, y, 'b-', label='Posição Y Real')
    axs[1].axhline(y=target_y, color='r', linestyle='--', label='Alvo Y')
    axs[1].set_ylabel('Posição Y (m)')
    axs[1].legend()
    axs[1].grid(True)

    # --- Eixo Z ---
    axs[2].plot(t, z, 'b-', label='Posição Z Real (Altitude)')
    axs[2].axhline(y=target_z, color='r', linestyle='--', label='Alvo Z')
    axs[2].set_ylabel('Posição Z (m)')
    axs[2].set_xlabel('Tempo (s)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    file_name = f'grafico_resposta_{file_tag}.png'
    plt.savefig(file_name)
    print(f'Gráficos de resposta salvos como: {os.path.abspath(file_name)}')
    plt.close(fig)


# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL (SINTONIA E EXECUÇÃO LQI)
# ==============================================================================
if __name__ == '__main__':
    os4 = Quadrotor()
    
    print("--- Simulador de Controle LQI de Quadricóptero ---")
    
    # --- Ponto 1: Criar o Controlador ---
    lqi = LQIController(os4)
    
    # --- Ponto 2: SINTONIA (Ajustar Q e R) ---
    # Esta é a sua nova "sintonia". Altere estes pesos para
    # mudar o comportamento do controlador.
    
    # --- ALTERAÇÃO APLICADA (Sintonia Q_p) ---
    # Pesos para os 12 estados originais (SINTONIA ATUALIZADA)
    Q_p = np.diag([
        1.0, 1.0, 1.0,   # Posição (x, y, z) - Penalidade BAIXA
        2.0, 2.0, 2.0,   # Velocidade (vx, vy, vz)
        1.0, 1.0, 1.0,   # Ângulos (phi, theta, psi)
        1.5, 1.5, 1.5    # Vel. Angular (p, q, r) - Penalidade ALTA para estabilidade
    ])
    
    # --- ALTERAÇÃO APLICADA (Sintonia Q_i) ---
    # Pesos para os 3 estados integrais (ix, iy, iz) (SINTONIA ATUALIZADA)
    # Ganhos integrais devem ser pequenos para evitar "windup"
    Q_i = np.diag([
        0.1, 0.1, 0.1
    ])
    
    # Combinar em uma matriz Q aumentada (15x15)
    Q_aug = np.zeros((15, 15))
    Q_aug[:12, :12] = Q_p
    Q_aug[12:, 12:] = Q_i
    
    # --- ALTERAÇÃO APLICADA (Sintonia R_aug) ---
    # R (4x4): Penalidade sobre o Esforço de Controle (SINTONIA ATUALIZADA)
    # Aumentamos R significativamente para "acalmar" o controlador
    R_aug = np.diag([
        5.0, # Empuxo
        10.0, # Torque de Roll
        10.0, # Torque de Pitch
        10.0  # Torque de Yaw
    ])
    
    # Calcular as matrizes de ganho K_p e K_i
    try:
        lqi.tune(Q_aug, R_aug)
    except Exception as e:
        print(f"\nERRO: Falha ao resolver a equação de Riccati.")
        print("Isto pode acontecer se o sistema aumentado não for controlável.")
        print(f"Detalhes: {e}")
        exit()

    # --- Ponto 3: Definir o Alvo e Executar ---
    target_x, target_y, target_z = 3.0, 3.0, 3.0
    target_psi = 0.0 # 0 radianos
    target_pos = [target_x, target_y, target_z, target_psi]
    
    duration = 10.0
    
    # Executa a simulação de controle LQI
    sol = run_lqi_simulation(os4, lqi, target_pos, duration)
    
    if sol:
        # --- Ponto 4: Visualizar Resultados ---
        file_tag = f"goto_{target_x}x_{target_y}y_{target_z}z_LQI"
        
        plot_responses(sol, target_pos, file_tag)
        
        animate_and_save_3d(sol, file_tag, duration, os4, target_pos)

        print("\nSimulação LQI concluída e salva como GIF e PNG.")
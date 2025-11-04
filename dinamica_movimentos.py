import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import os

class Quadrotor:
    """
    Classe que representa o modelo dinâmico de um quadricóptero.
    Os parâmetros são baseados na Tabela E.1 da tese de Bouabdallah.
    Inclui um modelo de contato com o solo.
    """
    def __init__(self):
        # Parâmetros físicos do quadricóptero OS4
        self.m = 0.650  # massa (kg)
        self.g = 9.81   # aceleração da gravidade (m/s^2)
        self.Ixx = 7.5e-3  # momento de inércia em x (kg.m^2)
        self.Iyy = 7.5e-3  # momento de inércia em y (kg.m^2)
        self.Izz = 1.3e-2  # momento de inércia em z (kg.m^2)
        self.l = 0.23   # distância do centro aos motores (m)
        self.b = 3.13e-5 # coeficiente de empuxo (N.s^2)
        self.d = 7.5e-7  # coeficiente de arrasto (N.m.s^2)
        
        # Parâmetros do modelo de contato com o solo (mola-amortecedor)
        self.ground_k = 500.0 # Rigidez da mola do solo (N/m)
        self.ground_c = 100.0 # Coeficiente de amortecimento do solo (N.s/m)

        # Estado inicial padrão (para a maioria das manobras)
        self.initial_state = np.zeros(12)

    def dynamics(self, t, state):
        """
        Define as equações diferenciais que governam a dinâmica do quadricóptero.
        """
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        T = self.b * (w1**2 + w2**2 + w3**2 + w4**2)
        tau_phi = self.l * self.b * (w4**2 - w2**2)
        tau_theta = self.l * self.b * (w1**2 - w3**2)
        tau_psi = self.d * (w1**2 - w2**2 + w3**2 - w4**2)
        
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
        
        thrust_world = R @ np.array([0, 0, T])

        ax = thrust_world[0] / self.m 
        ay = thrust_world[1] / self.m
        az = thrust_world[2] / self.m - self.g
        
        force_ground = 0
        if z < 0:
            force_ground = -self.ground_k * z - self.ground_c * vz
        az += force_ground / self.m

        p_dot = (tau_phi - q * r * (self.Izz - self.Iyy)) / self.Ixx
        q_dot = (tau_theta - p * r * (self.Ixx - self.Izz)) / self.Iyy
        r_dot = (tau_psi - p * q * (self.Iyy - self.Ixx)) / self.Izz
        
        T_euler = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        
        # Evitar divisão por zero em theta = +/- 90 graus
        if abs(np.cos(theta)) < 1e-6:
            euler_rates = np.zeros(3) # Ou uma aproximação apropriada
        else:
            euler_rates = T_euler @ np.array([p, q, r])

        phi_dot = euler_rates[0]
        theta_dot = euler_rates[1]
        psi_dot = euler_rates[2]

        return [vx, vy, vz, ax, ay, az, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]

def animate_and_save_3d(sol, movement_type, duration, quad_params):
    """
    Cria e salva uma animação 3D da trajetória do quadricóptero como um GIF.
    O quadricóptero é desenhado como um "X".
    """
    x_data, y_data, z_data = sol.y[0], sol.y[1], sol.y[2]
    phi_data, theta_data, psi_data = sol.y[6], sol.y[7], sol.y[8]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- INÍCIO DA MODIFICAÇÃO: Eixos Fixos ---
    # Define limites fixos para os eixos, de -5 a 5 metros.
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    # --- FIM DA MODIFICAÇÃO ---

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Animação 3D: {movement_type.replace("_", " ").title()}', fontsize=16)

    # Adiciona um plano para representar o chão
    # O plano agora será desenhado de -5 a 5, de acordo com os limites fixos
    X_ground, Y_ground = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    Z_ground = np.zeros_like(X_ground)
    ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.2, color='gray')

    trail, = ax.plot([], [], [], '-', lw=2, color='blue', label='Rastro da Trajetória')
    line1, = ax.plot([], [], [], '-', lw=3, color='red') 
    line2, = ax.plot([], [], [], '-', lw=3, color='red', label='Quadricóptero (X)') # Adiciona label a uma das linhas
    
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
        x_c, y_c, z_c = x_data[frame], y_data[frame], z_data[frame]
        phi, theta, psi = phi_data[frame], theta_data[frame], psi_data[frame]

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
        
        trail.set_data_3d(x_data[:frame+1], y_data[:frame+1], z_data[:frame+1])
        return trail, line1, line2

    num_frames = len(sol.t)
    fps = int(num_frames / duration)
    if fps < 15: fps = 15

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    
    file_name = f'animacao_3d_{movement_type}_quadX_escalaFixa.gif'
    print(f'Criando animação: {file_name}...')
    ani.save(file_name, writer='pillow', fps=fps)
    print(f'Animação salva como: {os.path.abspath(file_name)}')
    plt.close(fig)

def run_simulation(quad, duration, movement_type, initial_state=None):
    """
    Executa uma simulação e gera a animação 3D.
    """
    global w1, w2, w3, w4
    
    if initial_state is None:
        initial_state = quad.initial_state

    w_hover = np.sqrt(quad.m * quad.g / (4 * quad.b))
    
    delta_w_move = 50
    delta_w_down = 15
    movements = {
        'go_up':          [w_hover + delta_w_move] * 4,
        'go_down':        [w_hover - delta_w_down] * 4,
        'pitch_forward':  [w_hover + delta_w_move, w_hover, w_hover - delta_w_move, w_hover],
        'pitch_backward': [w_hover - delta_w_move, w_hover, w_hover + delta_w_move, w_hover],
        'roll_right':     [w_hover, w_hover - delta_w_move, w_hover, w_hover + delta_w_move],
        'roll_left':      [w_hover, w_hover + delta_w_move, w_hover, w_hover - delta_w_move],
        'yaw_right':      [w_hover + delta_w_move, w_hover - delta_w_move, w_hover + delta_w_move, w_hover - delta_w_move],
        'yaw_left':       [w_hover - delta_w_move, w_hover + delta_w_move, w_hover - delta_w_move, w_hover + delta_w_move],
    }
    
    w1, w2, w3, w4 = movements[movement_type]
    
    sol = solve_ivp(
        quad.dynamics, 
        [0, duration], 
        initial_state, 
        t_eval=np.linspace(0, duration, int(duration * 30))
    )
    
    animate_and_save_3d(sol, movement_type, duration, quad)

if __name__ == '__main__':
    os4 = Quadrotor()
    
    movements_from_ground = [
        'go_up', 'pitch_forward', 'pitch_backward', 
        'roll_right', 'roll_left', 'yaw_right', 'yaw_left'
    ]
    
    print("Iniciando simulações padrão a partir do chão...")
    for move in movements_from_ground:
        run_simulation(os4, duration=4, movement_type=move)
        
    print("\nIniciando simulação especial de descida controlada...")
    start_height = 5.0
    initial_state_down = np.zeros(12)
    initial_state_down[2] = start_height
    run_simulation(os4, duration=7, movement_type='go_down', initial_state=initial_state_down)
        
    print("\nTodas as simulações foram concluídas e salvas como GIFs.")

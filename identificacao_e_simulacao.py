import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal
import os


def generate_prbs(duration, dt, switch_interval, low_val, high_val):
    """Gera um sinal Pseudo-Random Binary Sequence (PRBS)."""
    t = np.arange(0, duration, dt)
    n_points = len(t)
    prbs_signal = np.zeros(n_points)
    
    current_val = high_val
    next_switch = switch_interval
    for i in range(n_points):
        if t[i] >= next_switch:
            current_val = low_val if current_val == high_val else high_val
            next_switch += switch_interval
        prbs_signal[i] = current_val
    return t, prbs_signal

def identify_motor_model():
    """Realiza o experimento de identificação do sistema do motor."""
    print("--- Iniciando Questão 4: Identificação do Modelo do Motor ---")
    

    K_true = 0.936
    tau_true = 0.178
    true_motor_model = signal.TransferFunction([K_true], [tau_true, 1])

    duration = 20
    dt = 0.01
    t, u_prbs = generate_prbs(duration, dt, switch_interval=0.5, low_val=200, high_val=300)


    _, y_measured, _ = signal.lsim(true_motor_model, U=u_prbs, T=t)

    noise = np.random.normal(0, 2, y_measured.shape)
    y_measured += noise
    
    y_dot = np.gradient(y_measured, dt)
    A = np.vstack([-y_dot, u_prbs]).T
    
    params, _, _, _ = np.linalg.lstsq(A, y_measured, rcond=None)
    tau_ident, K_ident = params[0], params[1]

    print(f"\nParâmetros Reais (da Tese): K = {K_true:.4f}, Tau = {tau_true:.4f}")
    print(f"Parâmetros Identificados:    K = {K_ident:.4f}, Tau = {tau_ident:.4f}\n")

    identified_motor_model = signal.TransferFunction([K_ident], [tau_ident, 1])
    _, y_ident, _ = signal.lsim(identified_motor_model, U=u_prbs, T=t)

    plt.figure(figsize=(12, 8))
    plt.plot(t, u_prbs, 'g--', label='Entrada PRBS (Comando u(t))', alpha=0.7)
    plt.plot(t, y_measured, 'b-', label='Saída "Medida" do Motor Real (y(t))')
    plt.plot(t, y_ident, 'r-.', label=f'Saída do Modelo Identificado (K={K_ident:.2f}, τ={tau_ident:.2f})')
    plt.title('Identificação do Sistema do Motor com Sinal PRBS')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade Angular (rad/s) / Comando')
    plt.legend()
    plt.grid(True)
    
    filename = "identificacao_motor.png"
    plt.savefig(filename)
    print(f"Gráfico de identificação salvo como: {os.path.abspath(filename)}")
    plt.close()
    
    return K_ident, tau_ident

class QuadrotorIdentified:
    """
    Classe do quadricóptero que incorpora a dinâmica do motor identificada.
    O estado agora inclui a velocidade de cada rotor.
    """
    def __init__(self, K_motor, tau_motor):
        self.m, self.g = 0.650, 9.81
        self.Ixx, self.Iyy, self.Izz = 7.5e-3, 7.5e-3, 1.3e-2
        self.l, self.b, self.d = 0.23, 3.13e-5, 7.5e-7

        self.K_motor = K_motor
        self.tau_motor = tau_motor

        self.initial_state = np.zeros(16)

    def dynamics(self, t, state):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4 = state
        
        global u1, u2, u3, u4

        w1_dot = (-w1 + self.K_motor * u1) / self.tau_motor
        w2_dot = (-w2 + self.K_motor * u2) / self.tau_motor
        w3_dot = (-w3 + self.K_motor * u3) / self.tau_motor
        w4_dot = (-w4 + self.K_motor * u4) / self.tau_motor

        T = self.b * (w1**2 + w2**2 + w3**2 + w4**2)
        tau_phi = self.l * self.b * (w4**2 - w2**2)
        tau_theta = self.l * self.b * (w1**2 - w3**2)
        tau_psi = self.d * (w1**2 - w2**2 + w3**2 - w4**2)
        
        ax = (T * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))) / self.m
        ay = (T * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))) / self.m
        az = (T * (np.cos(phi) * np.cos(theta))) / self.m - self.g
        
        p_dot = (tau_phi - q * r * (self.Izz - self.Iyy)) / self.Ixx
        q_dot = (tau_theta - p * r * (self.Ixx - self.Izz)) / self.Iyy
        r_dot = (tau_psi - p * q * (self.Iyy - self.Ixx)) / self.Izz
        
        phi_dot = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi) - r * np.sin(phi)
        psi_dot = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

        return [vx, vy, vz, ax, ay, az, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot,
                w1_dot, w2_dot, w3_dot, w4_dot]

def simulate_and_save_identified(quad, duration, movement_type):
    """
    Executa a simulação com o modelo de motor identificado e salva os gráficos.
    """
    global u1, u2, u3, u4
    
    w_hover = np.sqrt(quad.m * quad.g / (4 * quad.b))
    u_hover = w_hover / quad.K_motor 
    delta_u = 50 / quad.K_motor  

    movements = {
        'go_up': [u_hover + delta_u] * 4,
        'go_down': [u_hover - delta_u] * 4,
        'pitch_forward': [u_hover + delta_u, u_hover, u_hover - delta_u, u_hover],
        'pitch_backward': [u_hover - delta_u, u_hover, u_hover + delta_u, u_hover],
        'roll_right': [u_hover, u_hover - delta_u, u_hover, u_hover + delta_u],
        'roll_left': [u_hover, u_hover + delta_u, u_hover, u_hover - delta_u],
        'yaw_right': [u_hover + delta_u, u_hover - delta_u, u_hover + delta_u, u_hover - delta_u],
        'yaw_left': [u_hover - delta_u, u_hover + delta_u, u_hover - delta_u, u_hover + delta_u],
    }

    u1, u2, u3, u4 = movements[movement_type]

    sol = solve_ivp(quad.dynamics, [0, duration], quad.initial_state, t_eval=np.linspace(0, duration, 300))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    title = f'Simulação (Modelo Identificado): {movement_type.replace("_", " ").title()}'
    fig.suptitle(title, fontsize=16)

    axs[0, 0].plot(sol.t, sol.y[0], label='x'); axs[0, 0].plot(sol.t, sol.y[1], label='y'); axs[0, 0].plot(sol.t, sol.y[2], label='z')
    axs[0, 0].set_title('Posição (m)'); axs[0, 0].legend(); axs[0, 0].grid(True)
    axs[0, 1].plot(sol.t, sol.y[3], label='vx'); axs[0, 1].plot(sol.t, sol.y[4], label='vy'); axs[0, 1].plot(sol.t, sol.y[5], label='vz')
    axs[0, 1].set_title('Velocidade (m/s)'); axs[0, 1].legend(); axs[0, 1].grid(True)
    axs[1, 0].plot(sol.t, np.rad2deg(sol.y[6]), label='Roll (phi)'); axs[1, 0].plot(sol.t, np.rad2deg(sol.y[7]), label='Pitch (theta)'); axs[1, 0].plot(sol.t, np.rad2deg(sol.y[8]), label='Yaw (psi)')
    axs[1, 0].set_title('Ângulos de Euler (graus)'); axs[1, 0].legend(); axs[1, 0].grid(True)
    axs[1, 1].plot(sol.t, np.rad2deg(sol.y[9]), label='p'); axs[1, 1].plot(sol.t, np.rad2deg(sol.y[10]), label='q'); axs[1, 1].plot(sol.t, np.rad2deg(sol.y[11]), label='r')
    axs[1, 1].set_title('Velocidades Angulares (graus/s)'); axs[1, 1].legend(); axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    file_name = f'simulacao_identificada_{movement_type}.png'
    plt.savefig(file_name)
    plt.close(fig)
    print(f'Gráfico salvo como: {os.path.abspath(file_name)}')

if __name__ == '__main__':
    K_identificado, tau_identificado = identify_motor_model()

    print("\n--- Iniciando Questão 5: Simulação do Quadricóptero com Modelo Identificado ---")
    quad_com_motor_real = QuadrotorIdentified(K_motor=K_identificado, tau_motor=tau_identificado)
    simulation_time = 30
    
    movements_to_simulate = [
        'go_up', 'go_down', 'pitch_forward', 'pitch_backward', 
        'roll_right', 'roll_left', 'yaw_right', 'yaw_left'
    ]
    
    for move in movements_to_simulate:
        simulate_and_save_identified(quad_com_motor_real, simulation_time, move)
        
    print("\nSimulações com o modelo identificado foram concluídas e salvas.")
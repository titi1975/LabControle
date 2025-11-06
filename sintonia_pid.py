import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. COPIAS DAS CLASSES (do script principal)
# ==============================================================================

class PIDController:
    """
    Implementação de um controlador PID (Proporcional-Integral-Derivativo).
    """
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, anti_windup_limit=1.0):
        # ... (código da classe PIDController idêntico ao arquivo anterior) ...
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0
        self.anti_windup_limit = anti_windup_limit

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        P_term = self.Kp * error
        
        # Termo Integral (com anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.anti_windup_limit, self.anti_windup_limit)
        I_term = self.Ki * self.integral
        
        # Termo Derivativo
        derivative = (error - self.prev_error) / dt
        D_term = self.Kd * derivative
        
        # Atualiza o erro anterior
        self.prev_error = error
        
        # Saída total
        output = P_term + I_term + D_term
        return output

class Quadrotor:
    """
    Classe que representa o modelo dinâmico de um quadricóptero.
    (Copiada do script principal)
    """
    def __init__(self):
        # ... (código da classe Quadrotor idêntico ao arquivo anterior) ...
        self.m = 0.650
        self.g = 9.81
        self.Ixx = 7.5e-3
        self.Iyy = 7.5e-3
        self.Izz = 1.3e-2
        self.l = 0.23
        self.b = 3.13e-5
        self.d = 7.5e-7
        self.ground_k = 500.0
        self.ground_c = 100.0
        self.initial_state = np.zeros(12)

    def dynamics(self, t, state, inputs):
        # ... (código da função dynamics idêntico ao arquivo anterior) ...
        w1, w2, w3, w4 = inputs
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        T = self.b * (w1**2 + w2**2 + w3**2 + w4**2)
        tau_phi = self.l * self.b * (w4**2 - w2**2)
        tau_theta = self.l * self.b * (w1**2 - w3**2)
        tau_psi = self.d * (w1**2 - w2**2 + w3**2 - w4**2)
        
        R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        R = R_z @ R_y @ R_x
        
        thrust_world = R @ np.array([0, 0, T])
        gravity_world = np.array([0, 0, -self.m * self.g])
        force_total = thrust_world + gravity_world
        force_ground = 0
        if z < 0:
            force_ground = -self.ground_k * z - self.ground_c * vz
        force_total[2] += force_ground
        acceleration = force_total / self.m
        ax, ay, az = acceleration

        p_dot = (tau_phi - q * r * (self.Izz - self.Iyy)) / self.Ixx
        q_dot = (tau_theta - p * r * (self.Ixx - self.Izz)) / self.Iyy
        r_dot = (tau_psi - p * q * (self.Iyy - self.Ixx)) / self.Izz
        
        T_euler = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
        
        if abs(np.cos(theta)) < 1e-6:
            euler_rates = np.zeros(3)
        else:
            euler_rates = T_euler @ np.array([p, q, r])

        phi_dot, theta_dot, psi_dot = euler_rates
        return [vx, vy, vz, ax, ay, az, phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]

# ==============================================================================
# 2. NOVAS FERRAMENTAS DE ANÁLISE DE DESEMPENHO
# ==============================================================================

def analyze_performance(times, data, setpoint, tolerance_percent=0.05):
    """
    Calcula o sobressinal e o tempo de assentamento para um conjunto de dados.
    """
    if setpoint == 0: # Evita divisão por zero
        setpoint = 1e-6
        
    # --- Cálculo do Sobressinal (Overshoot) ---
    peak = np.max(data)
    overshoot_value = peak - setpoint
    if overshoot_value < 0:
         overshoot_value = 0.0 # Sem sobressinal
         
    overshoot_percent = (overshoot_value / abs(setpoint)) * 100
    
    # --- Cálculo do Tempo de Assentamento (Settling Time) ---
    tolerance_band = abs(setpoint * tolerance_percent)
    upper_bound = setpoint + tolerance_band
    lower_bound = setpoint - tolerance_band
    
    # Encontra o primeiro índice onde os dados estão fora da banda
    # (começando do fim e indo para o começo)
    indices_fora_da_banda = np.where((data > upper_bound) | (data < lower_bound))[0]
    
    if len(indices_fora_da_banda) == 0:
        # Nunca saiu da banda (ou começou nela)
        settling_time = 0.0
    else:
        # O tempo de assentamento é o tempo do *último* ponto que estava *fora* da banda
        last_out_index = indices_fora_da_banda[-1]
        if last_out_index == len(times) - 1:
            settling_time = float('inf') # Nunca assentou
        else:
            settling_time = times[last_out_index + 1]
            
    return {
        'sobressinal_perc': overshoot_percent,
        'tempo_assentamento_s': settling_time,
        'valor_pico': peak,
        'banda_sup': upper_bound,
        'banda_inf': lower_bound
    }


def plot_tune_results(sol, mode, setpoint):
    """
    Plota os resultados da simulação E as métricas de desempenho.
    """
    plt.figure(figsize=(14, 7))
    
    if mode == 'attitude':
        data = np.rad2deg(sol.y[7]) # Pitch (theta)
        label = 'Pitch (theta)'
        unit = 'graus'
        title = 'Sintonia de Atitude (Pitch)'
    elif mode == 'position_z':
        data = sol.y[2] # Altitude (Z)
        label = 'Altitude (Z)'
        unit = 'm'
        title = 'Sintonia de Posição (Altitude)'
    elif mode == 'position_x':
        data = sol.y[0] # Posição (X)
        label = 'Posição (X)'
        unit = 'm'
        title = 'Sintonia de Posição (X)'
        
    # Analisa o desempenho
    metrics = analyze_performance(sol.t, data, setpoint)
    
    # Plota os dados
    plt.plot(sol.t, data, label=f'{label} (Pico: {metrics["valor_pico"]:.2f} {unit})', lw=2)
    
    # Linha de Setpoint
    plt.axhline(setpoint, color='g', linestyle='--', label=f'Setpoint ({setpoint} {unit})')
    
    # Bandas de Assentamento (ex: 5%)
    plt.axhline(metrics['banda_sup'], color='r', linestyle=':', 
                label=f'Banda de Assentamento (+/- 5%)')
    plt.axhline(metrics['banda_inf'], color='r', linestyle=':')
    
    # Linha de Tempo de Assentamento
    if np.isfinite(metrics['tempo_assentamento_s']) and metrics['tempo_assentamento_s'] > 0:
        plt.axvline(metrics['tempo_assentamento_s'], color='purple', linestyle='--', 
                    label=f'T. Assentamento: {metrics["tempo_assentamento_s"]:.2f} s')

    # Título com as métricas
    title_metrics = (f"{title} | Ganhos: Kp={sol.gains['Kp']}, Ki={sol.gains['Ki']}, Kd={sol.gains['Kd']}\n"
                     f"Sobressinal: {metrics['sobressinal_perc']:.2f}% | "
                     f"T. Assentamento (5%): {metrics['tempo_assentamento_s']:.2f} s")
    plt.title(title_metrics)
    
    plt.ylabel(f"Posição ({unit})")
    plt.xlabel('Tempo (s)')
    plt.legend()
    plt.grid(True)
    print(f"Exibindo gráfico para {title}...")
    print("FECHE O GRÁFICO para continuar...")
    plt.show() # Bloqueia a execução


def run_test_simulation(quad, duration, dt, mode, test_gains, 
                        att_gains=None, z_gains=None):
    """
    Executa uma simulação com um conjunto completo de ganhos PID.
    """
    
    # --- Ponto 1: Definir Ganhos para o Teste ---
    # Define os ganhos para o controlador que estamos testando
    pid_test_gains = {**test_gains, 'anti_windup_limit': 999.0}

    # Padrões para ganhos não sintonizados (caso não sejam passados)
    default_att_gains = {'Kp': 0.1, 'Ki': 0.0, 'Kd': 0.05, 'anti_windup_limit': 1.0}
    default_z_gains = {'Kp': 30.0, 'Ki': 15.0, 'Kd': 20.0, 'anti_windup_limit': 15.0}
    default_xy_gains = {'Kp': 0.8, 'Ki': 0.2, 'Kd': 0.5, 'anti_windup_limit': np.deg2rad(15)}
    
    # --- Nova Lógica de Ganhos (Correção do Bug) ---
    
    # Ganhos de Atitude
    if mode == 'attitude':
        pid_att_gains = pid_test_gains
    elif att_gains: # Se estamos no modo z ou x E att_gains foi passado
        pid_att_gains = {**att_gains, 'anti_windup_limit': 1.0}
    else: # Fallback (embora não deva acontecer no fluxo normal)
        pid_att_gains = default_att_gains
        
    # Ganhos de Altitude (Z)
    if mode == 'position_z':
        pid_z_gains = pid_test_gains
    elif z_gains: # Se estamos no modo x E z_gains foi passado
        pid_z_gains = {**z_gains, 'anti_windup_limit': 15.0}
    else: # Modo 'attitude' ou fallback
        pid_z_gains = default_z_gains

    # Ganhos de Posição (XY)
    if mode == 'position_x':
        pid_xy_gains = pid_test_gains
    else: # Modo 'attitude' ou 'position_z'
        pid_xy_gains = default_xy_gains
    
    # Ganhos de Yaw (Sempre Padrão)
    pid_yaw_gains = {'Kp': 0.1, 'Ki': 0.01, 'Kd': 0.05, 'anti_windup_limit': 0.5}
    
    # --- Fim da Nova Lógica ---

    # --- Ponto 2: Definir Setpoints para o Teste ---
    if mode == 'attitude':
        sp_att = np.deg2rad(10.0) # Setpoint: 10 graus de pitch
        sp_z = 1.0 # Pairar a 1m
        sp_x = 0.0
    elif mode == 'position_z':
        sp_att = 0.0
        sp_z = 2.0 # Setpoint: 2 metros
        sp_x = 0.0
    elif mode == 'position_x':
        sp_att = 0.0 # Será definido pelo PID de X
        sp_z = 2.0 # Manter 2m
        sp_x = 2.0 # Setpoint: 2 metros X

    # --- Ponto 3: Inicialização dos Controladores ---
    pid_z = PIDController(setpoint=sp_z, **pid_z_gains)
    pid_x = PIDController(setpoint=sp_x, **pid_xy_gains)
    pid_y = PIDController(setpoint=0.0, **pid_xy_gains)
    
    pid_phi = PIDController(**pid_att_gains)
    pid_theta = PIDController(**pid_att_gains)
    pid_psi = PIDController(setpoint=0.0, **pid_yaw_gains)
    
    # --- Ponto 4: Misturador ---
    b, d, l = quad.b, quad.d, quad.l
    alloc_matrix = np.array([[b, b, b, b], [0, -l*b, 0, l*b], [l*b, 0, -l*b, 0], [d, -d, d, -d]])
    inv_alloc_matrix = np.linalg.inv(alloc_matrix)
    max_thrust_per_motor = (quad.m * quad.g * 2) / 4.0
    max_w_sq = max_thrust_per_motor / quad.b

    # --- Ponto 5: O Loop de Simulação ---
    state = quad.initial_state.copy()
    if mode == 'position_x':
        state[2] = sp_z # Começa no alto para o teste de X
    
    time = 0.0
    times = [time]
    states = [state]
    num_steps = int(duration / dt)

    for i in range(num_steps):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state
        
        thrust_ff = quad.m * quad.g # Feed-forward
        
        # 1. Loop Externo
        if mode == 'attitude':
            total_thrust = thrust_ff
            theta_des = sp_att
            phi_des = 0.0
        else:
            thrust_correction = pid_z.update(z, dt)
            total_thrust = thrust_ff + thrust_correction
            theta_des = pid_x.update(x, dt)
            phi_des = -pid_y.update(y, dt)

        # 2. Loop Interno
        pid_phi.set_setpoint(phi_des)
        pid_theta.set_setpoint(theta_des)
        tau_phi = pid_phi.update(phi, dt)
        tau_theta = pid_theta.update(theta, dt)
        tau_psi = pid_psi.update(psi, dt)

        # 3. Misturador
        control_vector = np.array([total_thrust, tau_phi, tau_theta, tau_psi])
        w_squared = inv_alloc_matrix @ control_vector
        w_squared = np.clip(w_squared, 0, max_w_sq)
        inputs = np.sqrt(w_squared)
        
        # 4. Dinâmica
        state_dot = quad.dynamics(time, state, inputs)
        state = state + np.array(state_dot) * dt
        
        # 5. Armazenamento
        time += dt
        times.append(time)
        states.append(state.copy())

    solution = type('Solution', (object,), {})()
    solution.t = np.array(times)
    solution.y = np.array(states).T 
    solution.gains = test_gains # Salva os ganhos usados para o título
    return solution

# ==============================================================================
# 3. SCRIPT DE EXECUÇÃO INTERATIVO (FOCO EM MÉTRICAS)
# ==============================================================================

def get_gains_from_user(current_gains):
    """Pede ao usuário os ganhos Kp, Ki, Kd."""
    try:
        kp = float(input(f"  Digite Kp (atual: {current_gains['Kp']}): "))
        ki = float(input(f"  Digite Ki (atual: {current_gains['Ki']}): "))
        kd = float(input(f"  Digite Kd (atual: {current_gains['Kd']}): "))
        return {'Kp': kp, 'Ki': ki, 'Kd': kd}
    except ValueError:
        print("Entrada inválida. Usando ganhos anteriores.")
        return current_gains

if __name__ == '__main__':
    os4 = Quadrotor()
    
    # Ganhos iniciais (pode ser da sua sintonia heurística ou Z-N)
    # Estes serão os valores que vamos refinar.
    gains_att = {'Kp': 0.35, 'Ki': 2.3, 'Kd': 0.03} # Z-N (Ku=0.35, Tu=0.3)
    gains_z   = {'Kp': 36.0, 'Ki': 48.0, 'Kd': 6.75} # Z-N (Ku=60, Tu=1.5)
    gains_xy  = {'Kp': 0.6, 'Ki': 0.57, 'Kd': 0.16}  # Z-N (Ku=1.0, Tu=2.1)
    
    print("--- Assistente Interativo de Sintonia (Foco em Métricas) ---")
    print("Você irá refinar Kp, Ki, Kd para cada loop de controle.")
    print("FECHE cada gráfico para prosseguir.")

    # --- ETAPA 1: SINTONIZAR ATITUDE (LOOP INTERNO) ---
    print("\n--- ETAPA 1: Sintonizando Loop Interno (Atitude: Pitch) ---")
    while True:
        gains_att = get_gains_from_user(gains_att)
        sol = run_test_simulation(os4, duration=5.0, dt=0.01, 
                                  mode='attitude', test_gains=gains_att)
        plot_tune_results(sol, 'attitude', setpoint=10.0)
        
        resp = input("Ganhos de ATITUDE estão bons? (s/n): ").lower()
        if resp == 's':
            print(f"Ganhos de ATITUDE finais: {gains_att}")
            break
        print("Ok, vamos tentar novos ganhos para Atitude.")

    # --- ETAPA 2: SINTONIZAR ALTITUDE (LOOP EXTERNO Z) ---
    print("\n--- ETAPA 2: Sintonizando Loop Externo (Posição: Altitude Z) ---")
    while True:
        gains_z = get_gains_from_user(gains_z)
        sol = run_test_simulation(os4, duration=8.0, dt=0.01, 
                                  mode='position_z', 
                                  test_gains=gains_z,
                                  att_gains=gains_att) # Usa ganhos da Etapa 1
        plot_tune_results(sol, 'position_z', setpoint=2.0)
        
        resp = input("Ganhos de ALTITUDE estão bons? (s/n): ").lower()
        if resp == 's':
            print(f"Ganhos de ALTITUDE finais: {gains_z}")
            break
        print("Ok, vamos tentar novos ganhos para Altitude.")

    # --- ETAPA 3: SINTONIZAR POSIÇÃO (LOOP EXTERNO X) ---
    print("\n--- ETAPA 3: Sintonizando Loop Externo (Posição: X) ---")
    while True:
        gains_xy = get_gains_from_user(gains_xy)
        sol = run_test_simulation(os4, duration=10.0, dt=0.01, 
                                  mode='position_x', 
                                  test_gains=gains_xy,
                                  att_gains=gains_att, # Usa ganhos da Etapa 1
                                  z_gains=gains_z)     # Usa ganhos da Etapa 2
        plot_tune_results(sol, 'position_x', setpoint=2.0)
        
        resp = input("Ganhos de POSIÇÃO X/Y estão bons? (s/n): ").lower()
        if resp == 's':
            print(f"Ganhos de POSIÇÃO X/Y finais: {gains_xy}")
            break
        print("Ok, vamos tentar novos ganhos para Posição X/Y.")

    # --- SAÍDA FINAL ---
    print("\n" + "="*60)
    print("SINTONIA FOCADA EM MÉTRICAS CONCLUÍDA. Copie o bloco abaixo:")
    print("="*60 + "\n")
    
    print("# --- GANHOS SINTONIZADOS MANUALMENTE (Foco em Métricas) ---")
    print(f"z_gains = {{**{gains_z}, 'anti_windup_limit': 15.0}}")
    print(f"xy_gains = {{**{gains_xy}, 'anti_windup_limit': np.deg2rad(15)}}")
    print(f"attitude_gains = {{**{gains_att}, 'anti_windup_limit': 1.0}}")
    print("# Ganhos de Yaw (não sintonizados, manter padrão)")
    print("yaw_gains = {'Kp': 0.1, 'Ki': 0.01, 'Kd': 0.05, 'anti_windup_limit': 0.5}")
    print("\n" + "="*60)
    print("Cole este bloco no seu script 'run_pid_simulation.py'")
    print("substituindo os ganhos 'heurísticos' ou Z-N antigos.")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURAÇÃO ---
NOME_ARQUIVO = 'teste.txt'  # Nome do seu arquivo txt

def analisar_motor():
    if not os.path.exists(NOME_ARQUIVO):
        print(f"ERRO: O arquivo '{NOME_ARQUIVO}' não foi encontrado na pasta.")
        print("Crie um arquivo txt com os dados copiados do Arduino.")
        return

    print(f"Lendo arquivo: {NOME_ARQUIVO}...")

    try:
        # 1. Tenta ler o arquivo normalmente
        df = pd.read_csv(NOME_ARQUIVO)
        
        # 2. Limpeza: Remove espaços em branco dos nomes das colunas e converte para MAIÚSCULO
        # Isso resolve o problema se você escreveu "pwm", "PWM ", " pwm" etc.
        df.columns = [c.strip().upper() for c in df.columns]

        # 3. Verificação de segurança:
        # Se a primeira linha era apenas números (ex: 0,0), o pandas usou os números como título.
        # Se 'PWM' não estiver nas colunas, recarregamos assumindo que NÃO tem cabeçalho.
        if 'PWM' not in df.columns:
            print("Aviso: Cabeçalho não detectado ou incorreto. Recarregando assumindo formato 'PWM,RPM'...")
            df = pd.read_csv(NOME_ARQUIVO, header=None, names=['PWM', 'RPM'])

        # 4. Limpeza de dados (Remove linhas com texto 'FIM' ou sujeira)
        df = df[pd.to_numeric(df['PWM'], errors='coerce').notnull()]
        
        # Converte para números (garantia final)
        df['PWM'] = df['PWM'].astype(float)
        df['RPM'] = df['RPM'].astype(float)

    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")
        return

    # --- LÓGICA DE ANÁLISE (Igual ao anterior) ---
    
    # Identificar Zona Morta (primeiro PWM onde RPM > 0)
    movimento = df[df['RPM'] > 0]
    
    if movimento.empty:
        print("ALERTA: O motor não registrou movimento (RPM=0 em todos os pontos).")
        return

    deadzone_pwm = movimento.iloc[0]['PWM']
    max_rpm = df['RPM'].max()
    
    print("-" * 30)
    print(f"RESULTADOS PARA {NOME_ARQUIVO}")
    print(f"-> Zona Morta (Deadzone): PWM {int(deadzone_pwm)}")
    print(f"-> RPM Máximo: {int(max_rpm)}")
    print("-" * 30)

    # Regressão Linear (y = mx + b)
    X = movimento['PWM'].values
    y = movimento['RPM'].values
    
    # Se tiver poucos pontos, não faz regressão para não dar erro
    if len(X) > 1:
        coef = np.polyfit(X, y, 1)
        poly1d_fn = np.poly1d(coef)
        equacao = f"RPM = {coef[0]:.2f} * PWM + {coef[1]:.2f}"
        print(f"Equação da Reta: {equacao}")
    else:
        poly1d_fn = None

    # --- GRÁFICO ---
    plt.figure(figsize=(10, 6))
    
    # Pontos reais
    plt.plot(df['PWM'], df['RPM'], 'o-', label='Medição Real', markersize=4, alpha=0.6)
    
    # Linha de tendência
    if poly1d_fn is not None:
        plt.plot(X, poly1d_fn(X), '--r', label='Tendência Linear', linewidth=2)

    plt.axvline(x=deadzone_pwm, color='green', linestyle=':', label=f'Início Movimento (PWM {int(deadzone_pwm)})')
    
    plt.title(f'Curva do Motor', fontsize=14)
    plt.xlabel('PWM (0-255)', fontsize=12)
    plt.ylabel('RPM', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    print("Gerando gráfico...")
    plt.show()

if __name__ == "__main__":
    analisar_motor()
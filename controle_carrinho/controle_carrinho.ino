#include <AFMotor.h>
#include <SoftwareSerial.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// =================================================================================
// --- CONFIGURAÇÕES FÍSICAS ---
// =================================================================================
const int FUROS_DISCO = 20;
const float RAIO_RODA_CM = 3.25; 
const float CIRCUNFERENCIA = 2 * PI * RAIO_RODA_CM;
const float DISTANCIA_POR_PULSO = CIRCUNFERENCIA / FUROS_DISCO;

// --- Configurações dos Controladores ---
const unsigned long INTERVALO_CONTROLE = 100; 
const float TOLERANCIA_ANGULO = 2.0;          
const float TOLERANCIA_DISTANCIA = 5.0;       

const float KP_DIST = 12.0;   
const float KP_HEAD = 6.0;    
const int VEL_MIN_MOVIMENTO = 110; 
const int VEL_GIRO_BASE = 150;     

// =================================================================================
// --- OBJETOS E PINOS ---
// =================================================================================
SoftwareSerial bluetoothSerial(9, 10); 
AF_DCMotor motor1(1); 
AF_DCMotor motor2(2); 
AF_DCMotor motor3(3); 
AF_DCMotor motor4(4); 
Adafruit_MPU6050 mpu;

const int pinSensor1 = A0;
const int pinSensor2 = A1;
const int pinSensor3 = A2;
const int pinSensor4 = A3;

// =================================================================================
// --- VARIÁVEIS GLOBAIS ---
// =================================================================================
float posX = 0.0;
float posY = 0.0;
float anguloZ = 0.0; 

float targetX = 0.0;
float targetY = 0.0;
float targetAngle = 0.0;

// Variável para rastrear estado manual (Correção do erro de compilação)
bool manualForward = false;

// Variáveis para RPM Manual
int targetRPM1=0, targetRPM2=0, targetRPM3=0, targetRPM4=0;
int pwm1=0, pwm2=0, pwm3=0, pwm4=0;
int currentRPM1=0, currentRPM2=0, currentRPM3=0, currentRPM4=0;

float gyroZ_offset = 0.0;
unsigned long previousMillisGyro = 0;

volatile unsigned long pulsos1=0, pulsos2=0, pulsos3=0, pulsos4=0;
int lastState1=0, lastState2=0, lastState3=0, lastState4=0;

// Modos de Operação
enum Modo { PARADO, MANUAL_SIMPLES, RPM_MANUAL, ANGULO_FIXO, POSICAO_XY };
Modo modoAtual = PARADO;

enum EstadoMovimento { GIRANDO_PARA_ALVO, AVANCANDO };
EstadoMovimento estadoPos = GIRANDO_PARA_ALVO;

unsigned long previousMillisControl = 0;

// =================================================================================
// --- SETUP ---
// =================================================================================
void setup() {
  Serial.begin(9600);
  pinMode(9, INPUT); 
  pinMode(10, OUTPUT);
  bluetoothSerial.begin(9600);
  
  pinMode(pinSensor1, INPUT); pinMode(pinSensor2, INPUT);
  pinMode(pinSensor3, INPUT); pinMode(pinSensor4, INPUT);

  Serial.println("--- INICIANDO ROBO V7 ---");
  bluetoothSerial.println("INICIANDO V7");

  if (!mpu.begin()) {
    Serial.println("ERRO: MPU6050");
    bluetoothSerial.println("ERRO: MPU6050");
    while(1); 
  }
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  Serial.println("Calibrando Gyro...");
  float soma = 0;
  for (int i = 0; i < 200; i++) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    soma += g.gyro.z;
    delay(5);
  }
  gyroZ_offset = soma / 200.0;
  
  Stop();
  Serial.println("SISTEMA PRONTO V7");
}

// =================================================================================
// --- LOOP PRINCIPAL ---
// =================================================================================
void loop() {
  if (bluetoothSerial.available() > 0) {
    String input = bluetoothSerial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      Serial.print("RX: "); Serial.println(input);
      processarComando(input);
    }
  }

  lerEncoder(pinSensor1, lastState1, pulsos1);
  lerEncoder(pinSensor2, lastState2, pulsos2);
  lerEncoder(pinSensor3, lastState3, pulsos3);
  lerEncoder(pinSensor4, lastState4, pulsos4);

  calcularAnguloAtual();

  unsigned long currentMillis = millis();
  if (currentMillis - previousMillisControl >= INTERVALO_CONTROLE) {
    
    atualizarOdometria();
    
    switch (modoAtual) {
      case POSICAO_XY:
        controlarPosicaoXY();
        break;
        
      case ANGULO_FIXO:
        controlarOrientacao(targetAngle);
        break;
      
      case RPM_MANUAL:
        controlarMotoresRPM();
        break;

      case MANUAL_SIMPLES:
        // No modo manual simples, os motores já foram setados no comando
        // Não fazemos nada aqui para manter o estado
        break;
        
      case PARADO:
      default:
        // Garante que pare se cair aqui
        break;
    }
    
    enviarDados(); 
    previousMillisControl = currentMillis;
  }
}

// =================================================================================
// --- LÓGICA DE CONTROLE POSICIONAL ---
// =================================================================================
void controlarPosicaoXY() {
  float dx = targetX - posX;
  float dy = targetY - posY;
  float distanciaAlvo = sqrt(dx*dx + dy*dy);
  
  if (distanciaAlvo < TOLERANCIA_DISTANCIA) {
    Stop();
    modoAtual = PARADO;
    Serial.println("ALVO ALCANCADO!");
    bluetoothSerial.println("FIM: ALVO OK");
    return;
  }

  float anguloDesejado = atan2(dy, dx) * 180.0 / PI;

  if (estadoPos == GIRANDO_PARA_ALVO) {
    float erroAng = anguloDesejado - anguloZ;
    while (erroAng > 180) erroAng -= 360;
    while (erroAng <= -180) erroAng += 360;

    if (abs(erroAng) <= TOLERANCIA_ANGULO) {
      Stop();
      delay(100);
      estadoPos = AVANCANDO;
    } else {
      int velGiro = VEL_GIRO_BASE + abs(erroAng) * 2;
      velGiro = constrain(velGiro, 0, 255);
      if (erroAng > 0) girarEsquerda(velGiro);
      else girarDireita(velGiro);
    }
  } 
  else if (estadoPos == AVANCANDO) {
    float erroAng = anguloDesejado - anguloZ;
    while (erroAng > 180) erroAng -= 360;
    while (erroAng <= -180) erroAng += 360;

    if (abs(erroAng) > 40) {
      estadoPos = GIRANDO_PARA_ALVO;
      return;
    }
    
    int velocidadeBase = distanciaAlvo * KP_DIST;
    velocidadeBase = constrain(velocidadeBase, VEL_MIN_MOVIMENTO, 255);
    
    int correcao = erroAng * KP_HEAD;
    
    // Baseado na lógica V5: M3/M4 Esquerda, M1/M2 Direita
    int motorEsq = velocidadeBase - correcao; // M3, M4
    int motorDir = velocidadeBase + correcao; // M1, M2
    
    motorEsq = constrain(motorEsq, 0, 255);
    motorDir = constrain(motorDir, 0, 255);
    
    moverFrenteDiferencial(motorEsq, motorDir);
  }
}

// =================================================================================
// --- PROCESSAMENTO DE COMANDOS ---
// =================================================================================
void processarComando(String cmd) {
  cmd.toUpperCase(); 

  // P:x,y (Posição)
  if (cmd.startsWith("P")) {
    int sep = cmd.indexOf(':');
    int comma = cmd.indexOf(',');
    if (sep != -1 && comma != -1) {
      targetX = cmd.substring(sep + 1, comma).toFloat();
      targetY = cmd.substring(comma + 1).toFloat();
      modoAtual = POSICAO_XY;
      estadoPos = GIRANDO_PARA_ALVO;
    }
  }
  // A:angulo (Orientação)
  else if (cmd.startsWith("A")) {
    int sep = cmd.indexOf(':');
    if (sep != -1) {
      targetAngle = cmd.substring(sep + 1).toFloat();
      modoAtual = ANGULO_FIXO;
    }
  }
  // M: Manual RPM
  else if (cmd.startsWith("M") && isDigit(cmd.charAt(1))) {
    modoAtual = RPM_MANUAL;
    int motorID = cmd.substring(1, 2).toInt();
    int val = cmd.substring(cmd.indexOf(':') + 1).toInt();
    if (motorID == 1) targetRPM1 = val;
    if (motorID == 2) targetRPM2 = val;
    if (motorID == 3) targetRPM3 = val;
    if (motorID == 4) targetRPM4 = val;
    controlarMotoresRPM();
  } 
  // COMANDOS MANUAIS CLÁSSICOS (Restaurados V5)
  else {
    char c = cmd.charAt(0);
    modoAtual = MANUAL_SIMPLES; // Define modo manual para não ser sobrescrito
    
    switch (c) {
      case 'F': forward(); break;
      case 'B': back(); break;
      case 'L': left(); break;
      case 'R': right(); break;
      case 'S': case '0': case 'D': 
        modoAtual = PARADO; 
        Stop(); 
        break;
    }
  }
}

// =================================================================================
// --- HARDWARE DE MOTORES (Mapeamento V5 Restaurado) ---
// =================================================================================

// Define velocidade global para todos
void setAllSpeed(int speed) {
  motor1.setSpeed(speed); motor2.setSpeed(speed);
  motor3.setSpeed(speed); motor4.setSpeed(speed);
}

// Avançar (Todos para frente)
void forward() { 
  manualForward = true; // Flag de estado para odometria
  setAllSpeed(255); 
  motor1.run(FORWARD); motor2.run(FORWARD); 
  motor3.run(FORWARD); motor4.run(FORWARD); 
}

// Recuar (Todos para trás)
void back() { 
  manualForward = false;
  setAllSpeed(255); 
  motor1.run(BACKWARD); motor2.run(BACKWARD); 
  motor3.run(BACKWARD); motor4.run(BACKWARD); 
}

// Esquerda Manual
void left() { 
  manualForward = false; // Girar não conta como avanço linear
  girarEsquerda(255); 
}

// Direita Manual
void right() { 
  manualForward = false;
  girarDireita(255); 
}

// Parar
void Stop() {
  manualForward = false;
  targetRPM1=0; targetRPM2=0; targetRPM3=0; targetRPM4=0;
  motor1.run(RELEASE); motor2.run(RELEASE); 
  motor3.run(RELEASE); motor4.run(RELEASE);
}

// --- FUNÇÕES DE GIRO (Mapeamento V5) ---
// Esquerda: Lado Dir (M1,M2) Frente | Lado Esq (M3,M4) Trás
void girarEsquerda(int spd) {
  motor1.setSpeed(spd); motor1.run(FORWARD);  
  motor2.setSpeed(spd); motor2.run(FORWARD); 
  
  motor3.setSpeed(spd); motor3.run(BACKWARD);  
  motor4.setSpeed(spd); motor4.run(BACKWARD); 
}

// Direita: Lado Dir (M1,M2) Trás | Lado Esq (M3,M4) Frente
void girarDireita(int spd) {
  motor1.setSpeed(spd); motor1.run(BACKWARD); 
  motor2.setSpeed(spd); motor2.run(BACKWARD);  
  
  motor3.setSpeed(spd); motor3.run(FORWARD); 
  motor4.setSpeed(spd); motor4.run(FORWARD);  
}

// --- CONTROLE DIFERENCIAL (Para uso no Controle Posicional) ---
// Mapeia Esquerda -> M3, M4 e Direita -> M1, M2
void moverFrenteDiferencial(int esq, int dir) {
  // Lado Esquerdo (M3, M4)
  motor3.setSpeed(esq); motor3.run(FORWARD);
  motor4.setSpeed(esq); motor4.run(FORWARD);

  // Lado Direito (M1, M2)
  motor1.setSpeed(dir); motor1.run(FORWARD);
  motor2.setSpeed(dir); motor2.run(FORWARD);
}

// =================================================================================
// --- OUTROS CONTROLES ---
// =================================================================================
void controlarOrientacao(float anguloAlvo) {
  float erro = anguloAlvo - anguloZ;
  while (erro > 180) erro -= 360;
  while (erro <= -180) erro += 360;

  if (abs(erro) <= TOLERANCIA_ANGULO) {
    Stop();
    return;
  }
  
  int spd = VEL_GIRO_BASE + abs(erro);
  spd = constrain(spd, 0, 255);
  
  // Como restauramos o giro V5, chamamos diretamente
  if (erro > 0) girarEsquerda(spd);
  else girarDireita(spd);
}

void controlarMotoresRPM() {
  ajustarMotor(motor1, pwm1, currentRPM1, targetRPM1);
  ajustarMotor(motor2, pwm2, currentRPM2, targetRPM2);
  ajustarMotor(motor3, pwm3, currentRPM3, targetRPM3);
  ajustarMotor(motor4, pwm4, currentRPM4, targetRPM4);
}

void ajustarMotor(AF_DCMotor &motor, int &pwmAtual, int rpmAtual, int rpmAlvo) {
  if (rpmAlvo == 0) { pwmAtual = 0; motor.run(RELEASE); return; }
  motor.setSpeed(rpmAlvo); motor.run(FORWARD);
}

// =================================================================================
// --- UTILITÁRIOS E TELEMETRIA ---
// =================================================================================
void atualizarOdometria() {
  // Cálculo de RPM instantâneo antes de zerar os pulsos
  // Fórmula: (Pulsos * 60000ms) / (Furos * Intervalo_ms)
  // Simplificando para Intervalo=100ms e Furos=20: Fator = 30
  unsigned long fator = 60000 / (FUROS_DISCO * INTERVALO_CONTROLE);
  
  currentRPM1 = pulsos1 * fator;
  currentRPM2 = pulsos2 * fator;
  currentRPM3 = pulsos3 * fator;
  currentRPM4 = pulsos4 * fator;

  long mediaPulsos = (pulsos1 + pulsos2 + pulsos3 + pulsos4) / 4;
  pulsos1 = 0; pulsos2 = 0; pulsos3 = 0; pulsos4 = 0;
  
  // CORRIGIDO: Usa a flag manualForward em vez de motor1.run
  bool movendoFrente = (modoAtual == POSICAO_XY && estadoPos == AVANCANDO) || 
                       (modoAtual == MANUAL_SIMPLES && manualForward); 

  if (mediaPulsos > 0) { 
     // Apenas atualiza se não estivermos no modo de giro explícito
     if (modoAtual != ANGULO_FIXO && estadoPos != GIRANDO_PARA_ALVO) {
        float distanciaDelta = mediaPulsos * DISTANCIA_POR_PULSO;
        float radianos = anguloZ * PI / 180.0;
        posX += distanciaDelta * cos(radianos);
        posY += distanciaDelta * sin(radianos);
     }
  }
}

void calcularAnguloAtual() {
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  unsigned long currentMillis = millis();
  float dt = (currentMillis - previousMillisGyro) / 1000.0;
  previousMillisGyro = currentMillis;

  float giroZ = (g.gyro.z - gyroZ_offset) * 57.2958;
  if (abs(giroZ) > 0.05) anguloZ += giroZ * dt;

  // --- NORMALIZAÇÃO ADICIONADA ---
  // Mantém o ângulo entre -360 e 360
  // Se passar de 360, vira 1, etc.
  if (anguloZ >= 360.0) anguloZ -= 360.0;
  else if (anguloZ <= -360.0) anguloZ += 360.0;
}

void lerEncoder(int pino, int &ultimoEstado, volatile unsigned long &contador) {
  int estadoAtual = digitalRead(pino);
  if (estadoAtual == HIGH && ultimoEstado == LOW) contador++;
  ultimoEstado = estadoAtual;
}

void enviarDados() {
  String msg = "POS:";
  msg += String(posX, 1); msg += ",";
  msg += String(posY, 1); 
  msg += "|ANG:";
  msg += String(anguloZ, 1);
  
  // Adiciona RPMs ao Bluetooth
  msg += "|RPM:";
  msg += currentRPM1; msg += ",";
  msg += currentRPM2; msg += ",";
  msg += currentRPM3; msg += ",";
  msg += currentRPM4;

  bluetoothSerial.println(msg);
  Serial.println(msg);
}
#include <AFMotor.h>
#include <SoftwareSerial.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// =================================================================================
// --- CONFIGURAÇÕES DE TUNING (AJUSTE FINO) ---
// =================================================================================

// FÍSICA
// Voltamos para 20 para ter melhor resolução de RPM.
// A correção de distância (x2) será feita na função atualizarOdometria.
const int FUROS_DISCO = 20; 
const float RAIO_RODA_CM = 3.25; 
const float CIRCUNFERENCIA = 2 * PI * RAIO_RODA_CM;
const float DISTANCIA_POR_PULSO = CIRCUNFERENCIA / FUROS_DISCO;

// TIMERS
// Aumentado para 100ms para garantir leitura estável de RPM em baixas velocidades
const unsigned long INTERVALO_CONTROLE = 100;  
const unsigned long INTERVALO_TELEMETRIA = 300;

// TOLERÂNCIAS
const float TOLERANCIA_ANGULO = 3.0;          
const float TOLERANCIA_DISTANCIA = 3.0;       

// LIMITES
const int RPM_MAX = 200; // Aumentado margem
const int PWM_MIN_MOVIMENTO = 60; 

// --- GANHOS PID ---

// PID DISTÂNCIA 
const float KP_DIST = 4.0;  
const float KI_DIST = 0.0;
const float KD_DIST = 0.0;

// PID ÂNGULO 
const float KP_HEAD = 5.0; 
const float KI_HEAD = 0.0; 
const float KD_HEAD = 0.1;

// PID MOTOR (RPM -> PWM)
// Ajustado para o novo intervalo de 100ms
const float KP_RPM = 1.0;   // Suavizado para evitar oscilação
const float KI_RPM = 0.8;   
const float KD_RPM = 0.0;
const float KF_RPM = 1.0;   // Feedforward

// Fator de Filtro (0.0 a 1.0). Quanto menor, mais suave (menos ruído), mas mais lento.
const float FILTRO_RPM = 0.6; 

// =================================================================================
// --- OBJETOS ---
// =================================================================================
SoftwareSerial bluetoothSerial(9, 10); 
AF_DCMotor motor1(1); AF_DCMotor motor2(2); 
AF_DCMotor motor3(3); AF_DCMotor motor4(4); 
Adafruit_MPU6050 mpu;

const int pinSensor1 = A0; const int pinSensor2 = A1;
const int pinSensor3 = A2; const int pinSensor4 = A3;

// =================================================================================
// --- ESTRUTURAS ---
// =================================================================================
struct PIDController {
    float Kp, Ki, Kd, Kf;
    float setpoint;
    float lastError;
    float integral;
    unsigned long lastTime;
    int stallBoost; 
};

void PID_Init(PIDController &pid, float kp, float ki, float kd, float kf) {
    pid.Kp = kp; pid.Ki = ki; pid.Kd = kd; pid.Kf = kf;
    pid.setpoint = 0.0; pid.lastError = 0.0; pid.integral = 0.0; pid.stallBoost = 0;
    pid.lastTime = millis();
}

// =================================================================================
// --- VARIÁVEIS ---
// =================================================================================
float posX = 0.0, posY = 0.0, anguloZ = 0.0; 
float gyroZ_offset = 0.0;
unsigned long previousMillisGyro = 0;

volatile unsigned long pulsos1=0, pulsos2=0, pulsos3=0, pulsos4=0;
int lastState1=0, lastState2=0, lastState3=0, lastState4=0;
bool manualForward = false; 

// Variáveis de RPM Filtradas
int rpmFilt1=0, rpmFilt2=0, rpmFilt3=0, rpmFilt4=0;

// Navegação
float targetX = 0.0, targetY = 0.0, targetAngle = 0.0;
int targetRPM1=0, targetRPM2=0, targetRPM3=0, targetRPM4=0; 
int pwm1=0, pwm2=0, pwm3=0, pwm4=0; 

// PIDs
PIDController pidDistancia, pidAngulo;
PIDController pidRPM1, pidRPM2, pidRPM3, pidRPM4;

enum Modo { PARADO, RPM_MANUAL, ANGULO_FIXO, POSICAO_XY };
Modo modoAtual = PARADO;

enum EstadoMovimento { GIRANDO_PARA_ALVO, AVANCANDO };
EstadoMovimento estadoPos = GIRANDO_PARA_ALVO;

unsigned long prevControlTime = 0;
unsigned long prevTelemTime = 0;

// =================================================================================
// --- SETUP ---
// =================================================================================
void setup() {
  Serial.begin(9600);
  pinMode(9, INPUT); pinMode(10, OUTPUT);
  bluetoothSerial.begin(9600);
  bluetoothSerial.setTimeout(10); 
  
  pinMode(pinSensor1, INPUT); pinMode(pinSensor2, INPUT);
  pinMode(pinSensor3, INPUT); pinMode(pinSensor4, INPUT);

  PID_Init(pidDistancia, KP_DIST, KI_DIST, KD_DIST, 0.0);
  PID_Init(pidAngulo, KP_HEAD, KI_HEAD, KD_HEAD, 0.0);
  
  PID_Init(pidRPM1, KP_RPM, KI_RPM, KD_RPM, KF_RPM);
  PID_Init(pidRPM2, KP_RPM, KI_RPM, KD_RPM, KF_RPM);
  PID_Init(pidRPM3, KP_RPM, KI_RPM, KD_RPM, KF_RPM);
  PID_Init(pidRPM4, KP_RPM, KI_RPM, KD_RPM, KF_RPM);
  
  if (!mpu.begin()) while(1);
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  calibrarGiroscopio();
  Stop();
  Serial.println("V10 ESTAVEL");
}

// =================================================================================
// --- LOOP ---
// =================================================================================
void loop() {
  if (bluetoothSerial.available()) {
    String input = bluetoothSerial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) processarComando(input);
  }

  // Leitura Sensores (Polling rápido)
  lerEncoder(pinSensor1, lastState1, pulsos1);
  lerEncoder(pinSensor2, lastState2, pulsos2);
  lerEncoder(pinSensor3, lastState3, pulsos3);
  lerEncoder(pinSensor4, lastState4, pulsos4);
  calcularAnguloAtual();

  unsigned long currentMillis = millis();

  if (currentMillis - prevControlTime >= INTERVALO_CONTROLE) {
    atualizarOdometria(); 
    
    switch (modoAtual) {
      case POSICAO_XY: controlarPosicaoXY(); controlarMotoresRPM(); break;
      case ANGULO_FIXO: controlarOrientacao(targetAngle); controlarMotoresRPM(); break;
      case RPM_MANUAL: controlarMotoresRPM(); break;
      default: Stop(); break;
    }
    prevControlTime = currentMillis;
  }

  if (currentMillis - prevTelemTime >= INTERVALO_TELEMETRIA) {
    enviarDados();
    prevTelemTime = currentMillis;
  }
}

// =================================================================================
// --- PID MOTOR ---
// =================================================================================
float PID_Compute_Motor(PIDController &pid, float inputRPM) {
    unsigned long now = millis();
    float timeChange = (float)(now - pid.lastTime) / 1000.0;
    if (timeChange <= 0) timeChange = 0.1; 
    
    float error = pid.setpoint - inputRPM;
    
    pid.integral += pid.Ki * error * timeChange;
    pid.integral = constrain(pid.integral, -255, 255); 
    
    float derivative = pid.Kd * (error - pid.lastError) / timeChange;
    
    float feedforward = pid.setpoint * pid.Kf;

    pid.lastError = error;
    pid.lastTime = now;
    
    // Anti-Stall: Se quer andar (setpoint > 10) e não anda (input < 5)
    if (abs(pid.setpoint) > 10 && abs(inputRPM) < 5) {
       if (pid.setpoint > 0) pid.stallBoost += 4; 
       else pid.stallBoost -= 4;
    } else {
       pid.stallBoost = pid.stallBoost * 0.5; // Decai boost quando move
    }
    pid.stallBoost = constrain(pid.stallBoost, -80, 80);

    float output = (pid.Kp * error) + pid.integral + derivative + feedforward + pid.stallBoost;
    return output;
}

void controlarMotoresRPM() {
  ajustarMotor(motor1, pwm1, rpmFilt1, targetRPM1, pidRPM1);
  ajustarMotor(motor2, pwm2, rpmFilt2, targetRPM2, pidRPM2);
  ajustarMotor(motor3, pwm3, rpmFilt3, targetRPM3, pidRPM3);
  ajustarMotor(motor4, pwm4, rpmFilt4, targetRPM4, pidRPM4);
}

void ajustarMotor(AF_DCMotor &motor, int &pwmOut, int rpmAtual, int rpmAlvo, PIDController &pid) {
  if (rpmAlvo == 0) {
    motor.run(RELEASE); pwmOut = 0;
    pid.integral = 0; pid.stallBoost = 0; pid.setpoint = 0;
    return;
  }

  pid.setpoint = rpmAlvo; 
  float outputRaw = PID_Compute_Motor(pid, abs(rpmAtual));
  
  int pwmFinal = abs((int)outputRaw);
  if (pwmFinal > 0 && pwmFinal < PWM_MIN_MOVIMENTO) pwmFinal = PWM_MIN_MOVIMENTO;
  pwmFinal = constrain(pwmFinal, 0, 255);
  
  pwmOut = pwmFinal; 
  motor.setSpeed(pwmFinal);
  
  if (rpmAlvo > 0) motor.run(FORWARD);
  else motor.run(BACKWARD);
}

// =================================================================================
// --- UTILITÁRIOS ---
// =================================================================================
void atualizarOdometria() {
  // Fator = 60000 / (FUROS * INTERVALO). 
  // Ex: 60000 / (20 * 100) = 30.
  // 1 pulso = 30 RPM. 2 pulsos = 60 RPM. Resolução aceitável.
  unsigned long fator = 60000 / (FUROS_DISCO * INTERVALO_CONTROLE);
  
  int rpmInst1 = pulsos1 * fator;
  int rpmInst2 = pulsos2 * fator;
  int rpmInst3 = pulsos3 * fator;
  int rpmInst4 = pulsos4 * fator;

  // Filtro Média Móvel Exponencial (Suaviza o ruído do sensor)
  rpmFilt1 = (FILTRO_RPM * rpmFilt1) + ((1.0 - FILTRO_RPM) * rpmInst1);
  rpmFilt2 = (FILTRO_RPM * rpmFilt2) + ((1.0 - FILTRO_RPM) * rpmInst2);
  rpmFilt3 = (FILTRO_RPM * rpmFilt3) + ((1.0 - FILTRO_RPM) * rpmInst3);
  rpmFilt4 = (FILTRO_RPM * rpmFilt4) + ((1.0 - FILTRO_RPM) * rpmInst4);

  long mediaPulsos = (pulsos1 + pulsos2 + pulsos3 + pulsos4) / 4;
  
  if (mediaPulsos > 0 && manualForward) { 
      // CORREÇÃO DE DISTÂNCIA: Multiplicamos por 2.0 pois você reportou metade da leitura
      float distanciaDelta = (mediaPulsos * DISTANCIA_POR_PULSO) * 2.0;
      float radianos = anguloZ * PI / 180.0;
      posX += distanciaDelta * cos(radianos);
      posY += distanciaDelta * sin(radianos);
  }
  
  pulsos1 = 0; pulsos2 = 0; pulsos3 = 0; pulsos4 = 0;
}

// =================================================================================
// --- COMANDOS ---
// =================================================================================
void processarComando(String cmd) {
  cmd.toUpperCase(); 

  if (cmd == "RESET") {
    resetSistema(); return;
  }

  // P:x,y (Posição)
  if (cmd.startsWith("P")) {
    int sep = cmd.indexOf(':');
    int comma = cmd.indexOf(',');
    if (sep != -1 && comma != -1) {
      targetX = cmd.substring(sep + 1, comma).toFloat();
      targetY = cmd.substring(comma + 1).toFloat();
      modoAtual = POSICAO_XY;
      estadoPos = GIRANDO_PARA_ALVO;
      bluetoothSerial.println("OK:POS");
    }
  }
  // A:angulo
  else if (cmd.startsWith("A")) {
    int sep = cmd.indexOf(':');
    if (sep != -1) {
      targetAngle = cmd.substring(sep + 1).toFloat();
      modoAtual = ANGULO_FIXO;
      bluetoothSerial.println("OK:ANG");
    }
  }
  // M: Manual RPM (M1:120)
  else if (cmd.startsWith("M") && isDigit(cmd.charAt(1))) {
    modoAtual = RPM_MANUAL;
    manualForward = true;
    int motorID = cmd.substring(1, 2).toInt();
    int val = cmd.substring(cmd.indexOf(':') + 1).toInt();
    
    // Limita ao máximo permitido
    val = constrain(val, -RPM_MAX, RPM_MAX);
    
    if (motorID == 1) targetRPM1 = val;
    if (motorID == 2) targetRPM2 = val;
    if (motorID == 3) targetRPM3 = val;
    if (motorID == 4) targetRPM4 = val;
  } 
  // Comandos F, B, L, R
  else {
    char c = cmd.charAt(0);
    modoAtual = RPM_MANUAL; 
    int defaultRPM = 120; // Velocidade padrão segura
    
    switch (c) {
      case 'F': 
        manualForward = true;
        targetRPM1=defaultRPM; targetRPM2=defaultRPM; targetRPM3=defaultRPM; targetRPM4=defaultRPM; 
        break;
      case 'B': 
        manualForward = false;
        targetRPM1=-defaultRPM; targetRPM2=-defaultRPM; targetRPM3=-defaultRPM; targetRPM4=-defaultRPM; 
        break;
      case 'L': 
        manualForward = false;
        targetRPM1=defaultRPM; targetRPM2=defaultRPM; targetRPM3=-defaultRPM; targetRPM4=-defaultRPM; 
        break;
      case 'R': 
        manualForward = false;
        targetRPM1=-defaultRPM; targetRPM2=-defaultRPM; targetRPM3=defaultRPM; targetRPM4=defaultRPM; 
        break;
      case 'S': case '0': case 'D': 
        modoAtual = PARADO; Stop(); break;
    }
  }
}

// =================================================================================
// --- NAVEGAÇÃO E OUTROS ---
// =================================================================================
void controlarPosicaoXY() {
  float dx = targetX - posX;
  float dy = targetY - posY;
  float distanciaAlvo = sqrt(dx*dx + dy*dy);
  
  if (distanciaAlvo < TOLERANCIA_DISTANCIA) {
    Stop(); modoAtual = PARADO; bluetoothSerial.println("FIM:OK"); return;
  }

  float anguloDesejado = atan2(dy, dx) * 180.0 / PI;
  float erroAng = anguloDesejado - anguloZ;
  while (erroAng > 180) erroAng -= 360;
  while (erroAng <= -180) erroAng += 360;

  if (estadoPos == GIRANDO_PARA_ALVO) {
    manualForward = false;
    if (abs(erroAng) <= TOLERANCIA_ANGULO) {
      Stop(); delay(50); estadoPos = AVANCANDO; return;
    }
    float pidOut = erroAng * KP_HEAD; 
    if (abs(pidOut) < 50) pidOut = (pidOut > 0) ? 50 : -50; 
    targetRPM1 = targetRPM2 = pidOut; targetRPM3 = targetRPM4 = -pidOut;
  } 
  else if (estadoPos == AVANCANDO) {
    manualForward = true;
    if (abs(erroAng) > 35) { estadoPos = GIRANDO_PARA_ALVO; return; }
    
    float velLinear = distanciaAlvo * KP_DIST;
    velLinear = constrain(velLinear, -RPM_MAX, RPM_MAX);
    float correcao = erroAng * KP_HEAD;
    
    targetRPM3 = targetRPM4 = velLinear - correcao; 
    targetRPM1 = targetRPM2 = velLinear + correcao; 
  }
}

void controlarOrientacao(float anguloAlvo) {
  manualForward = false;
  float erro = anguloAlvo - anguloZ;
  while (erro > 180) erro -= 360;
  while (erro <= -180) erro += 360;

  if (abs(erro) <= TOLERANCIA_ANGULO) { Stop(); return; }
  
  float giroRPM = erro * KP_HEAD;
  if (abs(giroRPM) < 50) giroRPM = (giroRPM > 0) ? 50 : -50; 
  targetRPM1 = targetRPM2 = giroRPM; targetRPM3 = targetRPM4 = -giroRPM;
}

void resetSistema() {
  Stop(); posX = 0.0; posY = 0.0; anguloZ = 0.0; targetX = 0.0; targetY = 0.0; targetAngle = 0.0;
  PID_Init(pidDistancia, KP_DIST, KI_DIST, KD_DIST, 0.0);
  PID_Init(pidAngulo, KP_HEAD, KI_HEAD, KD_HEAD, 0.0);
  calibrarGiroscopio();
  bluetoothSerial.println("RESET:OK");
}

void calibrarGiroscopio() {
  float soma = 0;
  for (int i = 0; i < 100; i++) {
    sensors_event_t a, g, t; mpu.getEvent(&a, &g, &t);
    soma += g.gyro.z; delay(5);
  }
  gyroZ_offset = soma / 100.0;
}

void Stop() {
  manualForward = false;
  targetRPM1=0; targetRPM2=0; targetRPM3=0; targetRPM4=0;
  pwm1=0; pwm2=0; pwm3=0; pwm4=0;
  pidRPM1.stallBoost = 0; pidRPM1.integral = 0;
  pidRPM2.stallBoost = 0; pidRPM2.integral = 0;
  pidRPM3.stallBoost = 0; pidRPM3.integral = 0;
  pidRPM4.stallBoost = 0; pidRPM4.integral = 0;
  motor1.run(RELEASE); motor2.run(RELEASE); motor3.run(RELEASE); motor4.run(RELEASE);
}

void calcularAnguloAtual() {
  sensors_event_t a, g, t; mpu.getEvent(&a, &g, &t);
  unsigned long currentMillis = millis();
  float dt = (currentMillis - previousMillisGyro) / 1000.0;
  previousMillisGyro = currentMillis;
  float giroZ = (g.gyro.z - gyroZ_offset) * 57.2958; 
  if (abs(giroZ) > 0.08) anguloZ += giroZ * dt; 
  if (anguloZ >= 360.0) anguloZ -= 360.0; else if (anguloZ <= -360.0) anguloZ += 360.0;
}

void lerEncoder(int pino, int &ultimoEstado, volatile unsigned long &contador) {
  int estadoAtual = digitalRead(pino);
  if (estadoAtual == HIGH && ultimoEstado == LOW) contador++;
  ultimoEstado = estadoAtual;
}

void enviarDados() {
  String msg = "P:"; msg += String(posX, 0); msg += ","; msg += String(posY, 0); 
  msg += "|A:"; msg += String(anguloZ, 0);
  // Envia RPM Filtrado e PWM para debug
  msg += "|M1:"; msg += rpmFilt1; msg += " ("; msg += pwm1; msg += ")";
  msg += "|M2:"; msg += rpmFilt2;
  msg += "|M3:"; msg += rpmFilt3;
  msg += "|M4:"; msg += rpmFilt4;
  bluetoothSerial.println(msg);
}
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

// TIMERS
// Aumentado para 200ms para dobrar a precisão de leitura em baixas rotações
const unsigned long INTERVALO_ODOMETRIA = 200;  
const unsigned long INTERVALO_TELEMETRIA = 300; 

// Filtro suavizador (0.7 mantem mais histórico, suavizando mais)
const float FILTRO_RPM = 0.6; 

// Configuração do Teste de Mínimo (Ramp Down)
const int PWM_INICIAL_TESTE = 180; // Começa com força suficiente
const int DEGRAU_PWM = 5;          // Reduz 5 de PWM por vez
const int INTERVALO_DEGRAU = 500;  // A cada 0.5 segundos

// =================================================================================
// --- OBJETOS ---
// =================================================================================
SoftwareSerial bluetoothSerial(9, 10); 
AF_DCMotor motor1(1); AF_DCMotor motor2(2); 
AF_DCMotor motor3(3); AF_DCMotor motor4(4); 
Adafruit_MPU6050 mpu;

// --- PINAGEM ---
const int pinSensor1 = A0; 
const int pinSensor2 = A1;
const int pinSensor3 = A2; 
const int pinSensor4 = A3;

// =================================================================================
// --- VARIÁVEIS ---
// =================================================================================
float posX = 0.0, posY = 0.0, anguloZ = 0.0; 
float gyroZ_offset = 0.0;
unsigned long previousMillisGyro = 0;

volatile unsigned long pulsos1=0, pulsos2=0, pulsos3=0, pulsos4=0;
int lastState1=0, lastState2=0, lastState3=0, lastState4=0;

// Estado
bool movendoFrente = false; 
bool testandoMinimo = false; // Flag para o novo modo de teste

// Telemetria
int rpmFilt1=0, rpmFilt2=0, rpmFilt3=0, rpmFilt4=0;
int pwmAtual = 0; 

unsigned long prevOdomTime = 0;
unsigned long prevTelemTime = 0;
unsigned long prevRampTime = 0; // Timer para redução de velocidade

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
  
  if (!mpu.begin()) {
    Serial.println("ERRO MPU");
    while(1);
  }
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  calibrarGiroscopio();
  Stop();
  Serial.println("MODO TESTE: PRECISAO + RAMP DOWN");
}

// =================================================================================
// --- LOOP ---
// =================================================================================
void loop() {
  // 1. Comandos
  if (bluetoothSerial.available()) {
    String input = bluetoothSerial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) processarComando(input);
  }

  // 2. Sensores
  lerEncoder(pinSensor1, lastState1, pulsos1);
  lerEncoder(pinSensor2, lastState2, pulsos2);
  lerEncoder(pinSensor3, lastState3, pulsos3);
  lerEncoder(pinSensor4, lastState4, pulsos4);
  calcularAnguloAtual();

  unsigned long currentMillis = millis();

  // 3. Lógica do Teste F:MIN (Redução Gradual)
  if (testandoMinimo && (currentMillis - prevRampTime >= INTERVALO_DEGRAU)) {
    pwmAtual -= DEGRAU_PWM;
    
    if (pwmAtual <= 0) {
      pwmAtual = 0;
      testandoMinimo = false;
      Stop();
      bluetoothSerial.println("TESTE:FIM");
    } else {
      // Aplica o novo PWM reduzido
      aplicarMotores(pwmAtual, true); // true = frente
    }
    prevRampTime = currentMillis;
  }

  // 4. Odometria
  if (currentMillis - prevOdomTime >= INTERVALO_ODOMETRIA) {
    atualizarOdometria(); 
    prevOdomTime = currentMillis;
  }

  // 5. Telemetria
  if (currentMillis - prevTelemTime >= INTERVALO_TELEMETRIA) {
    enviarDados();
    prevTelemTime = currentMillis;
  }
}

// =================================================================================
// --- PROCESSAMENTO DE COMANDOS ---
// =================================================================================
void processarComando(String cmd) {
  cmd.toUpperCase(); 

  // Novo Comando de Teste de Mínimo
  if (cmd == "F:MIN") {
    testandoMinimo = true;
    movendoFrente = true;
    pwmAtual = PWM_INICIAL_TESTE;
    aplicarMotores(pwmAtual, true);
    prevRampTime = millis();
    bluetoothSerial.println("TESTE:INICIO");
    return;
  }

  if (cmd == "RESET") {
    resetSistema();
    return;
  }
  
  // Comandos Padrão
  char c = cmd.charAt(0);
  
  // Se receber qualquer outro comando manual, cancela o teste automático
  if (testandoMinimo) testandoMinimo = false;

  switch (c) {
    case 'F': // FRENTE MAXIMA
      movendoFrente = true;
      pwmAtual = 255;
      aplicarMotores(255, true);
      break;

    case 'L': // GIRO ESQUERDA MAXIMO
      movendoFrente = false;
      pwmAtual = 255;
      motor1.setSpeed(255); motor1.run(FORWARD);
      motor2.setSpeed(255); motor2.run(FORWARD);
      motor3.setSpeed(255); motor3.run(BACKWARD);
      motor4.setSpeed(255); motor4.run(BACKWARD);
      break;

    case 'R': // GIRO DIREITA MAXIMO
      movendoFrente = false;
      pwmAtual = 255;
      motor1.setSpeed(255); motor1.run(BACKWARD);
      motor2.setSpeed(255); motor2.run(BACKWARD);
      motor3.setSpeed(255); motor3.run(FORWARD);
      motor4.setSpeed(255); motor4.run(FORWARD);
      break;

    case 'S': // PARAR
    case '0':
    case 'D':
      Stop();
      break;
  }
}

// =================================================================================
// --- AUXILIARES ---
// =================================================================================

void aplicarMotores(int pwm, bool frente) {
  motor1.setSpeed(pwm); 
  motor2.setSpeed(pwm); 
  motor3.setSpeed(pwm); 
  motor4.setSpeed(pwm);
  
  if (frente) {
    motor1.run(FORWARD); motor2.run(FORWARD);
    motor3.run(FORWARD); motor4.run(FORWARD);
  } else {
    // Caso precise implementar ré no futuro
    motor1.run(BACKWARD); motor2.run(BACKWARD);
    motor3.run(BACKWARD); motor4.run(BACKWARD);
  }
}

void Stop() {
  testandoMinimo = false;
  movendoFrente = false;
  pwmAtual = 0;
  motor1.run(RELEASE); 
  motor2.run(RELEASE); 
  motor3.run(RELEASE); 
  motor4.run(RELEASE);
}

void resetSistema() {
  Stop(); 
  posX = 0.0; posY = 0.0; anguloZ = 0.0; 
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

void atualizarOdometria() {
  unsigned long fator = 60000 / (FUROS_DISCO * INTERVALO_ODOMETRIA);
  
  int rpmInst1 = pulsos1 * fator;
  int rpmInst2 = pulsos2 * fator;
  int rpmInst3 = pulsos3 * fator;
  int rpmInst4 = pulsos4 * fator;

  // Filtro
  rpmFilt1 = (FILTRO_RPM * rpmFilt1) + ((1.0 - FILTRO_RPM) * rpmInst1);
  rpmFilt2 = (FILTRO_RPM * rpmFilt2) + ((1.0 - FILTRO_RPM) * rpmInst2);
  rpmFilt3 = (FILTRO_RPM * rpmFilt3) + ((1.0 - FILTRO_RPM) * rpmInst3);
  rpmFilt4 = (FILTRO_RPM * rpmFilt4) + ((1.0 - FILTRO_RPM) * rpmInst4);

  long mediaPulsos = (pulsos1 + pulsos2 + pulsos3 + pulsos4) / 4;
  
  if (mediaPulsos > 0 && movendoFrente) { 
      float distanciaDelta = (mediaPulsos * DISTANCIA_POR_PULSO);
      float radianos = anguloZ * PI / 180.0;
      posX += distanciaDelta * cos(radianos);
      posY += distanciaDelta * sin(radianos);
  }
  
  pulsos1 = 0; pulsos2 = 0; pulsos3 = 0; pulsos4 = 0;
}

void enviarDados() {
  String msg = "P:"; msg += String(posX, 0); msg += ","; msg += String(posY, 0); 
  msg += "|A:"; msg += String(anguloZ, 0);
  msg += "|M1:"; msg += rpmFilt1; 
  msg += "|M2:"; msg += rpmFilt2;
  msg += "|M3:"; msg += rpmFilt3;
  msg += "|M4:"; msg += rpmFilt4;
  msg += "|PWM:"; msg += pwmAtual; 
  bluetoothSerial.println(msg);
}
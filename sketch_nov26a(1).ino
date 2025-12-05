#include <AFMotor.h>
#include <SoftwareSerial.h>

// Configuração do Bluetooth (RX no A0, TX no A1)
SoftwareSerial bluetooth(A0, A1); 

AF_DCMotor motor1(1);
AF_DCMotor motor2(2);
AF_DCMotor motor3(3);
AF_DCMotor motor4(4);

char comando;
int velocidade = 200;

void setup() {
  Serial.begin(9600);
  bluetooth.begin(9600); 
  
  // Esta mensagem prova que o código novo foi carregado:
  Serial.println("MODO BLUETOOTH ATIVADO: Aguardando app..."); 

  motor1.setSpeed(velocidade);
  motor2.setSpeed(velocidade);
  motor3.setSpeed(velocidade);
  motor4.setSpeed(velocidade);
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
}

void loop() {
  // O código só entra aqui se o celular mandar algo
  if (bluetooth.available() > 0) {
    comando = bluetooth.read();
    
    // Debug: Mostra no PC o que o celular mandou
    Serial.print("Comando recebido: ");
    Serial.println(comando); 

    if (comando == 'F') { // Frente
      motor1.run(FORWARD); motor2.run(FORWARD); motor3.run(FORWARD); motor4.run(FORWARD);
    }
    else if (comando == 'B') { // Ré
      motor1.run(BACKWARD); motor2.run(BACKWARD); motor3.run(BACKWARD); motor4.run(BACKWARD);
    }
    else if (comando == 'L') { // Esquerda
      motor1.run(BACKWARD); motor4.run(BACKWARD); motor2.run(FORWARD); motor3.run(FORWARD);
    }
    else if (comando == 'R') { // Direita
      motor1.run(FORWARD); motor4.run(FORWARD); motor2.run(BACKWARD); motor3.run(BACKWARD);
    }
    else if (comando == 'S') { // Parar
      motor1.run(RELEASE); motor2.run(RELEASE); motor3.run(RELEASE); motor4.run(RELEASE);
    }
    // Comandos de velocidade
    else if (comando >= '0' && comando <= '9') {
      velocidade = map(comando - '0', 0, 9, 0, 255);
      motor1.setSpeed(velocidade); motor2.setSpeed(velocidade);
      motor3.setSpeed(velocidade); motor4.setSpeed(velocidade);
    }
    else if (comando == 'q') {
      velocidade = 255;
      motor1.setSpeed(velocidade); motor2.setSpeed(velocidade);
      motor3.setSpeed(velocidade); motor4.setSpeed(velocidade);
    }
  }
}
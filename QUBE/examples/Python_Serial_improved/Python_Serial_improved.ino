#include "QUBE.hpp"

QUBE qube;

// Control parameters - can be updated by Python
float motor_voltage = 0.0;
float max_voltage = 8.0;
bool resetMotorEnc = false;
bool resetPendulumEnc = false;
int r = 0, g = 0, b = 999; // Default blue LED

// Timing variables
unsigned long last_control_time = 0;
const unsigned long CONTROL_INTERVAL_US = 500; // 2000Hz control loop (500µs)
unsigned long last_communication_time = 0;
const unsigned long COMM_INTERVAL_US = 10000; // 100Hz communication (10ms)

void setup() {
  Serial.begin(115200);
  qube.begin();
  qube.resetMotorEncoder();
  qube.resetPendulumEncoder();
  qube.setMotorVoltage(0);
  qube.setRGB(0, 0, 999); // Blue LED
  qube.update();
}

void checkForCommands() {
  // Process any incoming data without blocking
  if (Serial.available() >= 10) {
    resetMotorEnc = Serial.read();
    resetPendulumEnc = Serial.read();

    int r_MSB = Serial.read();
    int r_LSB = Serial.read();
    r = (r_MSB << 8) + r_LSB;

    int g_MSB = Serial.read();
    int g_LSB = Serial.read();
    g = (g_MSB << 8) + g_LSB;

    int b_MSB = Serial.read();
    int b_LSB = Serial.read();
    b = (b_MSB << 8) + b_LSB;

    int voltage_MSB = Serial.read();
    int voltage_LSB = Serial.read();
    int voltage_raw = (voltage_MSB << 8) + voltage_LSB - 999;
    motor_voltage = voltage_raw / 100.0; // Convert to float (-9.99 to 9.99)

    // Apply limits
    if (motor_voltage > max_voltage) motor_voltage = max_voltage;
    if (motor_voltage < -max_voltage) motor_voltage = -max_voltage;
  }
}

void sendData() {
  sendEncoderData(0); // Motor
  sendEncoderData(1); // Pendulum
  sendRPMData();
  sendCurrentData();
}

void sendEncoderData(bool encoder) {
  float encoderAngle = 0;

  if (encoder == 1) {
    encoderAngle = qube.getPendulumAngle(false);
  } else {
    encoderAngle = qube.getMotorAngle(false);
  }
  
  // Ensure consistent encoding matches what Python expects
  long revolutions = (long)(encoderAngle/360.0);
  float _angle = encoderAngle - revolutions*360.0;
  if (_angle < 0) _angle += 360.0; // Keep angle portion positive [0-360)
  
  long angle = (long)_angle;
  long angleDecimal = abs((_angle - angle) * 100);  // Use abs to ensure positive decimal

  // Add sign bit to revolutions if needed
  if (encoderAngle < 0) { 
    revolutions = abs(revolutions);
    revolutions |= (1<<15);  // Set high bit for negative
  }
  
  // Pack angle and decimal into one 16-bit value
  angle = (angle << 7) | (angleDecimal & 0x7F);  // Ensure decimal only uses 7 bits

  Serial.write(highByte(revolutions));
  Serial.write(lowByte(revolutions));
  Serial.write(highByte(angle));
  Serial.write(lowByte(angle));
}

void sendRPMData() {
  long rpm = (long)qube.getRPM();
  bool dir = rpm < 0;

  if (dir) {
    rpm = abs(rpm);
    rpm |= 1 << 15;
  }

  Serial.write(highByte(rpm));
  Serial.write(lowByte(rpm));
}

void sendCurrentData() {
  long current = (long)qube.getMotorCurrent();
  current = abs(current);
  Serial.write(highByte(current));
  Serial.write(lowByte(current));
}

void performControlUpdate() {
  // Apply resets if requested
  if (resetMotorEnc) {
    qube.resetMotorEncoder();
    resetMotorEnc = false;
  }
  
  if (resetPendulumEnc) {
    qube.resetPendulumEncoder();
    resetPendulumEnc = false;
  }

  // Apply control values
  qube.setRGB(r, g, b);
  qube.setMotorVoltage(motor_voltage);
  qube.update();
}

void loop() {
  unsigned long current_time = micros();
  
  // High-frequency control loop (2000Hz)
  if ((current_time - last_control_time) >= CONTROL_INTERVAL_US) {
    performControlUpdate();
    last_control_time = current_time;
  }
  
  // Lower frequency communication loop (100Hz)
  if ((current_time - last_communication_time) >= COMM_INTERVAL_US) {
    checkForCommands();
    sendData();
    last_communication_time = current_time;
  }
}
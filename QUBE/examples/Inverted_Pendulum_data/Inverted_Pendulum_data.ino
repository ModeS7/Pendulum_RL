#include "QUBE.hpp"

QUBE qube;

// Physical parameters
float m_p = 0.1; // Pendulum stick - kg
float l = 0.095; // Length of pendulum - meters
float l_com = l/2; // Distance to COM - meters
float J = (1/3) * m_p * l * l; // Inertia kg/m^2
float g = 9.81; // Gravitational constant - m/s^2

// Swingup Parameters
float Er = 0.015; // Reference energy - Joules
float ke = 50; // Tunable gain for swingup voltage - m/s/J
float u_max = 3; // m/s^2
float balance_range = 35.0; // Range where mode switches to balancing - degrees

// Balance Parameters
float s = 0.33; // Quick scaling
float kp_theta = 2 * s; // Kp pendulum angle
float kd_theta = 0.125 * s; // Kd pendulum angle
float kp_pos = 0.07 * s; // Kp motor angle
float kd_pos = 0.06 * s; // Kd motor angle
long t_balance = 0;

// Filter parameters
float twopi = 3.141592*2; // 2pi
float y_k_last = 0; // Pendulum angular velocity
float wc = 500/twopi; // Cutoff frequency
float y2_k_last = 0; // Motor angular velocity
float wc2 = 500/twopi; // Cutoff frequency
float y3_k_last = 0; // Voltage 
float wc3 = 500/twopi; // Cutoff frequency

// Control loop parameters
float freq = 2000; // Frequency - Hz
float dt = 1.0/freq; // Timestep - sec

// Program variables
float prevAngle = 0;
float prevPos = 0;
long last = micros();
long now = micros();
long t_reset = micros();
bool mode = 0;
bool lastMode = 0;
bool reset = false;

// Data collection variables
float voltage = 0;        // Applied voltage
float angularV = 0;       // Pendulum angular velocity
float motorV = 0;         // Motor angular velocity
float energy = 0;         // System energy
unsigned long startTime = 0; // Time when logging started
long dataTimestamp = 0;   // Time elapsed since start

// Data collection settings
bool dataLoggingEnabled = true;
int decimationFactor = 20; // Only log every Nth sample to avoid overwhelming serial
int sampleCounter = 0;

void setup() {
  Serial.begin(115200);
  qube.begin();
  delay(1000);
  qube.setRGB(999, 999, 999);
  qube.resetMotorEncoder();
  delay(1000);
  qube.setPendulumEncoder(0);
  qube.update();
  delay(1000);
  
  // Print CSV header
  Serial.println("time,position,angle,motor_velocity,pendulum_velocity,voltage,energy,mode");
  startTime = millis();
}

void sendDataToSerial(long timestamp, float position, float angle, float motorVelocity, 
                     float pendulumVelocity, float appliedVoltage, float systemEnergy, int controlMode) {
  if (!dataLoggingEnabled) return;
  
  // Only log every Nth sample to avoid overwhelming serial connection
  sampleCounter++;
  if (sampleCounter % decimationFactor != 0) return;
  
  // Write the data as CSV to serial
  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(position);
  Serial.print(",");
  Serial.print(angle);
  Serial.print(",");
  Serial.print(motorVelocity);
  Serial.print(",");
  Serial.print(pendulumVelocity);
  Serial.print(",");
  Serial.print(appliedVoltage);
  Serial.print(",");
  Serial.print(systemEnergy);
  Serial.print(",");
  Serial.println(controlMode);
}

float swingup(float angle) {
  angularV = (angle-prevAngle) / dt;
  prevAngle = angle;

  energy = 0.5 * J * angularV*angularV + m_p*g*l_com*(1-cos(angle));
  float u = ke * (energy - Er) * (-angularV * cos(angle));
  float u_sat = min(u_max, max(-u_max, u));

  voltage = u_sat * (8.4*0.095*0.085) /0.042;

  // Low pass filter voltage to reduce noise
  float y3_k = y3_k_last + wc3*dt*(voltage - y3_k_last);
  y3_k_last = y3_k;

  qube.setMotorVoltage(y3_k);
  return y3_k; // Return the applied voltage
}

float settlePendulum(float angle) {
  // Hack for using energy equation to make target energy 0
  if (angle < 0) {
    angle += 180;
    angle = 180 - angle;
  } else {
    angle -= 180;
    angle = -180 - angle;
  }

  angularV = (angle-prevAngle) / dt;
  prevAngle = angle;

  energy = 0.5 * J * angularV*angularV + m_p*g*l_com*(1-cos(angle));
  float u = -ke * (energy - Er) * (angularV * cos(angle));
  float u_sat = min(u_max, max(-u_max, u));

  voltage = u_sat * (8.4*0.095*0.085) /0.042;

  // Low pass filter the voltage to reduce noise
  float y3_k = y3_k_last + wc3*dt*(voltage - y3_k_last);
  y3_k_last = y3_k;
  qube.setMotorVoltage(y3_k);
  return y3_k;
}

float settleMotor(float position) {
  motorV = (position-prevPos) / dt;
  prevPos = position;
  float y2_k = y2_k_last + wc2*dt*(motorV - y2_k_last);
  y2_k_last = y2_k;
  
  float u_pos = kp_pos * 3 * position + kd_pos * y2_k;
  qube.setMotorVoltage(-u_pos);
  return -u_pos;
}

float balance(float position, float angle) {
  angularV = (angle-prevAngle) / dt;
  motorV = (position-prevPos) / dt;

  // Low pass filtering velocities
  float y_k = y_k_last + wc*dt*(angularV - y_k_last);
  float y2_k = y2_k_last + wc2*dt*(motorV - y2_k_last);

  float u_ang = kp_theta * angle + kd_theta * y_k;
  float u_pos = kp_pos * position + kd_pos * y2_k;
  float u = u_pos + u_ang;
  qube.setMotorVoltage(u);
  prevAngle = angle;
  prevPos = position;
  y_k_last = y_k;
  y2_k_last = y2_k;
  
  // Calculate energy for data collection
  energy = 0.5 * J * angularV*angularV + m_p*g*l_com*(1-cos(angle));
  
  return u; // Return the applied voltage
}

void loop() {
  now = micros();
  if (now - last < dt*1e6) return;
  
  float appliedVoltage = 0.0;
  dataTimestamp = millis() - startTime; // Time in milliseconds since logging started
  
  last = now;
  int position = qube.getMotorAngle();
  float angle = qube.getPendulumAngle();
  angle > 0 ? angle -= 180 : angle += 180; // Makes 0 degrees be the top position

  if (mode == 0 && angle < (balance_range) && angle > (-balance_range)) {
    mode = 1;
    t_balance = now;
  }
  if (mode == 1 && !(angle < (balance_range) && angle > (-balance_range))) {
    mode = 0;
    if (now - t_balance > 1e6) {
      reset = true;
      t_reset = now;
    }
  }

  // Reset
  if (reset) {
    while (micros() - t_reset < 2e6) {
      appliedVoltage = settleMotor(position);
      motorV = (position-prevPos) / dt;
      
      // Collect data during reset phase too (with mode = 2 for reset)
      dataTimestamp = millis() - startTime;
      sendDataToSerial(dataTimestamp, position, angle, motorV, angularV, appliedVoltage, 0, 2);
      
      qube.update();
      position = qube.getMotorAngle();
      delay(dt*1e3);
    }
    reset = false;
    return;
  }

  if (mode == 0) appliedVoltage = swingup(angle);
  if (mode == 1) appliedVoltage = balance(position, angle);

  // Send data to serial
  sendDataToSerial(dataTimestamp, position, angle, motorV, angularV, appliedVoltage, energy, mode);
  
  qube.update();
  lastMode = mode;
}
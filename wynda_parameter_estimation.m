%% Motor-driven pendulum parameter estimation using WyNDA
% Based on the paper "Discovering State-Space Representation of 
% Dynamical Systems From Noisy Data" by Agus Hasan

clear;
clc;

%% Pendulum system parameters (true values)
% Motor parameters
Rm = 8.94;  % Motor resistance (Ohm)
Km = 0.0431;  % Motor back-emf constant
Jm = 6e-5;  % Total moment of inertia on motor shaft (kg·m^2)
bm = 3e-4;  % Viscous damping coefficient (Nm/rad/s)

% Pendulum parameters
DA = 3e-4;  % Damping coefficient of pendulum arm (Nm/rad/s)
DL = 5e-4;  % Damping coefficient of pendulum link (Nm/rad/s)
mA = 0.053;  % Weight of pendulum arm (kg)
mL = 0.024;  % Weight of pendulum link (kg)
LA = 0.086;  % Length of pendulum arm (m)
LL = 0.128;  % Length of pendulum link (m)
JA = 5.72e-5;  % Inertia moment of pendulum arm (kg·m^2)
JL = 1.31e-4;  % Inertia moment of pendulum link (kg·m^2)
g = 9.81;  % Gravity constant (m/s^2)

% Pre-compute constants
half_mL_LL_g = 0.5 * mL * LL * g;
half_mL_LL_LA = 0.5 * mL * LL * LA;
quarter_mL_LL_squared = 0.25 * mL * LL^2;
Mp_g_Lp = mL * g * LL;
Jp = (1/3) * mL * LL^2;  % Pendulum moment of inertia

% Time settings
t_end = 10;  % Simulation time (s)
dt = 0.001;  % Time step
N = t_end / dt;  % Number of iterations
t = 0:dt:t_end;  % Time array

%% Number of variables and coefficients
n = 4;  % Number of states [theta_m, theta_L, theta_m_dot, theta_L_dot]
r = 16 * n;  % Number of parameters (16 basis functions per state)

% Measurement matrix
C = eye(n);  % Full state measurement

%% Initial conditions
x = [0.1; 0; 0; 0];  % Initial state [theta_m, theta_L, theta_m_dot, theta_L_dot]
xbar = x;  % Initial state estimate (AO)
xhat = x;  % Initial state estimate (AKF)
y = x;  % Initial measurement
thetabar = zeros(r,1);  % Initial parameter estimate (AO)
thetahat = thetabar;  % Initial parameter estimate (AKF)
u = 0;  % Initial control input

% Preallocate arrays for efficiency
x_store = zeros(n, N+1);
x_store(:, 1) = x; 
xbar_store = zeros(n, N+1);
xbar_store(:, 1) = xbar;
y_store = zeros(n, N+1);
y_store(:, 1) = y;
thetabar_store = zeros(r, N+1);
thetabar_store(:, 1) = thetabar;
xhat_store = zeros(n, N+1);
xhat_store(:, 1) = xbar;
thetahat_store = zeros(r, N+1);
thetahat_store(:, 1) = thetahat;
u_store = zeros(1, N+1);
u_store(:, 1) = u;

%% Initialization for estimator 
% Observer parameters (AO)
lambdav = 0.999;
lambdat = 0.999;
Rx = 0.1 * eye(n);
Rt = 1 * eye(n);
Px = 100 * eye(n);
Pt = 10 * eye(r);
Gamma = zeros(n, r);

% Kalman Filter parameters (AKF)
P = 1000000 * eye(n);
Rk = 1 * eye(n);
Qk = 0.1 * eye(n);
Upsilon = zeros(n, r);
S = 100000 * eye(r);
lambda = 0.99999;
Alpha = 0.999;

%% Simulation
for i = 1:N
    % Control input (sinusoidal for good excitation)
    u = 2 * sin(t(i)*5);
    
    % Get current state
    theta_m = x(1);
    theta_L = x(2);
    theta_m_dot = x(3);
    theta_L_dot = x(4);
    
    % Calculate motor torque
    im = (u - Km * theta_m_dot) / Rm;
    Tm = Km * im;
    
    % Calculate dynamics coefficients
    % For theta_m equation:
    M11 = mA * LA^2 + quarter_mL_LL_squared - quarter_mL_LL_squared * cos(theta_L)^2 + JA;
    M12 = -half_mL_LL_LA * cos(theta_L);
    C1 = 0.5 * mL * LL^2 * sin(theta_L) * cos(theta_L) * theta_m_dot * theta_L_dot;
    C2 = half_mL_LL_LA * sin(theta_L) * theta_L_dot^2;
    
    % For theta_L equation:
    M21 = -half_mL_LL_LA * cos(theta_L);
    M22 = JL + quarter_mL_LL_squared;
    C3 = -quarter_mL_LL_squared * cos(theta_L) * sin(theta_L) * theta_m_dot^2;
    G = half_mL_LL_g * sin(theta_L);
    
    % Calculate determinant
    det_M = M11 * M22 - M12 * M21;
    
    % Solve for accelerations
    if abs(det_M) < 1e-10
        theta_m_ddot = 0;
        theta_L_ddot = 0;
    else
        % Right-hand side of equations
        RHS1 = Tm - C1 - C2 - DA * theta_m_dot;
        RHS2 = -G - DL * theta_L_dot - C3;
        
        % Solve for accelerations
        theta_m_ddot = (M22 * RHS1 - M12 * RHS2) / det_M;
        theta_L_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M;
    end
    
    % Euler integration
    x = [x(1) + x(3) * dt;
         x(2) + x(4) * dt;
         x(3) + theta_m_ddot * dt;
         x(4) + theta_L_ddot * dt];
    
    % Add noise to measurements
    noise_level = 0.001;
    y = C * x + noise_level * randn(n, 1);
    
    % Define the basis functions for WyNDA
    % Create expanded representation with various nonlinear terms
    psi = [y(1), y(2), y(3), y(4), y(1)^2, y(2)^2, y(3)^2, y(4)^2, ...
           sin(y(1)), sin(y(2)), cos(y(1)), cos(y(2)), ...
           y(1)*y(2), y(3)*y(4), sin(y(1)+y(2)), u];
    
    % Build the block diagonal matrix of basis functions
    Psi = [psi, zeros(1,r/n*(n-1));
           zeros(1,r/n), psi, zeros(1, r/n*(n-2));
           zeros(1,r/n*2), psi, zeros(1, r/n);
           zeros(1,r/n*3), psi];
    
    % Estimation using Adaptive Observer (AO)
    % Prediction
    Kx = Px * C' / (C * Px * C' + Rx);
    Kt = Pt * Gamma' * C' / (C * Gamma * Pt * Gamma' * C' + Rt);
    Gamma = (eye(n) - Kx * C) * Gamma;
    xbar = xbar + (Kx + Gamma * Kt) * (y - C * xbar);
    thetabar = thetabar - Kt * (y - C * xbar);
    
    % Update
    xbar = xbar + Psi * thetabar;
    Px = (1 / lambdav) * eye(n) * (eye(n) - Kx * C) * Px * eye(n);
    Pt = (1 / lambdat) * (eye(r) - Kt * C * Gamma) * Pt;
    Gamma = eye(n) * Gamma - Psi;
    
    % Store values
    x_store(:, i+1) = x;
    xbar_store(:, i+1) = xbar;
    y_store(:, i+1) = y;
    thetabar_store(:, i+1) = thetabar;
    u_store(:, i+1) = u;
    
    % Estimation using Adaptive Kalman Filter (AKF)
    P = eye(n) * P * eye(n)' + Qk;
    Sigma = C * P * C' + Rk;
    K = P / Sigma;
    
    Omega = C * eye(n) * Upsilon + C * Psi;
    Upsilon = (eye(n) - K * C) * Upsilon + (eye(n) - K * C) * Psi;
    Lambda = pinv(lambda * Sigma + Omega * S * Omega');
    Pi = S * Omega' * Lambda;
    S = S / lambda - S / lambda * Omega' * Lambda * Omega * S;
    
    ytilde = y - (xhat + Psi * thetahat);
    Qk = Alpha * Qk + (1 - Alpha) * (K * (ytilde * ytilde') * K');
    Rk = Alpha * Rk + (1 - Alpha) * (ytilde * ytilde' + P);
    P = (eye(n) - K * C) * P;
    
    thetahat = thetahat + Pi * (y - C * xhat);
    xhat = eye(n) * xhat + Psi * thetahat + K * (y - C * xhat) + Upsilon * Pi * (y - C * xhat);
    
    % Store values
    xhat_store(:, i+1) = xhat;
    thetahat_store(:, i+1) = thetahat;
end

%% Parameter mapping
% Based on our understanding of the system dynamics, we can map parameters to physical values
% The identified parameters need to be scaled by 1/dt to convert to continuous time

% Extract and scale key identified parameters from AO
thetabar_scaled = thetabar_store(:, end) / dt;

% 1) Extract corresponding parameters based on the structure of our basis functions
% Note: These indices correspond to specific terms in the basis functions
% that relate to physical parameters (may need adjustment based on your specific implementation)

% Example extraction (adjust indices based on your specific implementation)
% Motor parameters (AO)
Rm_est_AO = thetabar_scaled(16) / thetabar_scaled(3);  % Motor resistance
Km_est_AO = thetabar_scaled(15);  % Motor constant
Jm_est_AO = 1 / thetabar_scaled(19);  % Motor inertia

% Pendulum parameters (AO)
mL_est_AO = thetabar_scaled(32) / thetabar_scaled(10) * 2 / g;  % Link mass
LL_est_AO = sqrt(thetabar_scaled(40) / mL_est_AO * 3);  % Link length

% Extract and scale key identified parameters from AKF
thetahat_scaled = thetahat_store(:, end) / dt;

% Motor parameters (AKF)
Rm_est_AKF = thetahat_scaled(16) / thetahat_scaled(3);  % Motor resistance
Km_est_AKF = thetahat_scaled(15);  % Motor constant
Jm_est_AKF = 1 / thetahat_scaled(19);  % Motor inertia

% Pendulum parameters (AKF)
mL_est_AKF = thetahat_scaled(32) / thetahat_scaled(10) * 2 / g;  % Link mass
LL_est_AKF = sqrt(thetahat_scaled(40) / mL_est_AKF * 3);  % Link length

%% Result Display
fprintf('=== Parameter Estimation Results ===\n');
fprintf('Parameter\tTrue Value\tAO Estimate\tAKF Estimate\n');
fprintf('Rm\t\t%.4f\t\t%.4f\t\t%.4f\n', Rm, Rm_est_AO, Rm_est_AKF);
fprintf('Km\t\t%.4f\t\t%.4f\t\t%.4f\n', Km, Km_est_AO, Km_est_AKF);
fprintf('Jm\t\t%.6f\t\t%.6f\t\t%.6f\n', Jm, Jm_est_AO, Jm_est_AKF);
fprintf('mL\t\t%.4f\t\t%.4f\t\t%.4f\n', mL, mL_est_AO, mL_est_AKF);
fprintf('LL\t\t%.4f\t\t%.4f\t\t%.4f\n', LL, LL_est_AO, LL_est_AKF);

%% Plotting Results
% State Trajectories
figure(1);
clf;
subplot(2,2,1);
plot(t, y_store(1,:), '-', 'LineWidth', 2);
hold on;
plot(t, xbar_store(1,:), ':', 'LineWidth', 2);
plot(t, xhat_store(1,:), ':', 'LineWidth', 2);
legend('measured', 'AO', 'AKF');
title('Motor Angle (\theta_m)');
xlabel('Time (s)');
grid on;

subplot(2,2,2);
plot(t, y_store(2,:), '-', 'LineWidth', 2);
hold on;
plot(t, xbar_store(2,:), ':', 'LineWidth', 2);
plot(t, xhat_store(2,:), ':', 'LineWidth', 2);
legend('measured', 'AO', 'AKF');
title('Link Angle (\theta_L)');
xlabel('Time (s)');
grid on;

subplot(2,2,3);
plot(t, y_store(3,:), '-', 'LineWidth', 2);
hold on;
plot(t, xbar_store(3,:), ':', 'LineWidth', 2);
plot(t, xhat_store(3,:), ':', 'LineWidth', 2);
legend('measured', 'AO', 'AKF');
title('Motor Angular Velocity (\theta_m\_dot)');
xlabel('Time (s)');
grid on;

subplot(2,2,4);
plot(t, y_store(4,:), '-', 'LineWidth', 2);
hold on;
plot(t, xbar_store(4,:), ':', 'LineWidth', 2);
plot(t, xhat_store(4,:), ':', 'LineWidth', 2);
legend('measured', 'AO', 'AKF');
title('Link Angular Velocity (\theta_L\_dot)');
xlabel('Time (s)');
grid on;

% Control Input
figure(2);
clf;
plot(t, u_store, 'LineWidth', 2);
title('Control Input');
xlabel('Time (s)');
ylabel('Voltage (V)');
grid on;

% Parameter Convergence
figure(3);
clf;
subplot(3,2,1);
plot(t, Rm*ones(1,length(t)), '-', 'LineWidth', 2);
hold on;
plot(t, thetabar_store(16,:)./thetabar_store(3,:)/dt, ':', 'LineWidth', 2);
plot(t, thetahat_store(16,:)./thetahat_store(3,:)/dt, ':', 'LineWidth', 2);
legend('true', 'AO', 'AKF');
title('Motor Resistance (R_m)');
xlabel('Time (s)');
grid on;

subplot(3,2,2);
plot(t, Km*ones(1,length(t)), '-', 'LineWidth', 2);
hold on;
plot(t, thetabar_store(15,:)/dt, ':', 'LineWidth', 2);
plot(t, thetahat_store(15,:)/dt, ':', 'LineWidth', 2);
legend('true', 'AO', 'AKF');
title('Motor Constant (K_m)');
xlabel('Time (s)');
grid on;

subplot(3,2,3);
plot(t, Jm*ones(1,length(t)), '-', 'LineWidth', 2);
hold on;
plot(t, 1./(thetabar_store(19,:)/dt), ':', 'LineWidth', 2);
plot(t, 1./(thetahat_store(19,:)/dt), ':', 'LineWidth', 2);
legend('true', 'AO', 'AKF');
title('Motor Inertia (J_m)');
xlabel('Time (s)');
grid on;

subplot(3,2,4);
plot(t, mL*ones(1,length(t)), '-', 'LineWidth', 2);
hold on;
plot(t, (thetabar_store(32,:)./thetabar_store(10,:)*2/g)/dt, ':', 'LineWidth', 2);
plot(t, (thetahat_store(32,:)./thetahat_store(10,:)*2/g)/dt, ':', 'LineWidth', 2);
legend('true', 'AO', 'AKF');
title('Link Mass (m_L)');
xlabel('Time (s)');
grid on;

subplot(3,2,5);
plot(t, LL*ones(1,length(t)), '-', 'LineWidth', 2);
hold on;
plot(t, sqrt(thetabar_store(40,:)./((thetabar_store(32,:)./thetabar_store(10,:)*2/g)/dt)*3), ':', 'LineWidth', 2);
plot(t, sqrt(thetahat_store(40,:)./((thetahat_store(32,:)./thetahat_store(10,:)*2/g)/dt)*3), ':', 'LineWidth', 2);
legend('true', 'AO', 'AKF');
title('Link Length (L_L)');
xlabel('Time (s)');
grid on;

% Root mean square error calculation
RMSE_AO = sqrt(mean((Rm - (thetabar_store(16,end)./thetabar_store(3,end)/dt))^2 + ...
                     (Km - (thetabar_store(15,end)/dt))^2 + ...
                     (Jm - (1/(thetabar_store(19,end)/dt)))^2 + ...
                     (mL - ((thetabar_store(32,end)./thetabar_store(10,end)*2/g)/dt))^2 + ...
                     (LL - sqrt(thetabar_store(40,end)./((thetabar_store(32,end)./thetabar_store(10,end)*2/g)/dt)*3))^2));

RMSE_AKF = sqrt(mean((Rm - (thetahat_store(16,end)./thetahat_store(3,end)/dt))^2 + ...
                      (Km - (thetahat_store(15,end)/dt))^2 + ...
                      (Jm - (1/(thetahat_store(19,end)/dt)))^2 + ...
                      (mL - ((thetahat_store(32,end)./thetahat_store(10,end)*2/g)/dt))^2 + ...
                      (LL - sqrt(thetahat_store(40,end)./((thetahat_store(32,end)./thetahat_store(10,end)*2/g)/dt)*3))^2));

fprintf('\nRoot Mean Square Error:\n');
fprintf('AO: %.6f\n', RMSE_AO);
fprintf('AKF: %.6f\n', RMSE_AKF);
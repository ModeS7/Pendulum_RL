%% Data-Driven Pendulum System Using Estimated Parameters
clear;
clc;

%% Learned Parameters from AKF
% Store the learned parameters in a vector (80 parameters total)
theta = zeros(80, 1);

% State 1 Parameters (theta_m, parameters 1-20)
theta(1:20) = [
    0.000136, -1.184000, -0.808862, -1.818295, -0.007482, 
    0.005216, 0.492130, -0.031609, 0.157202, -0.003850, 
    0.021640, -0.010877, 0.125459, -0.199537, 0.004575, 
    0.162773, 1.720716, -0.007264, -1.981190, 4.656705
];

% State 2 Parameters (theta_L, parameters 21-40)
theta(21:40) = [
    0.000119, -1.219403, -0.868572, -1.864837, -0.007233, 
    0.004246, 0.476013, -0.031857, 0.169125, -0.001998, 
    0.021827, -0.013101, 0.131768, -0.240857, 0.004186, 
    0.143614, 1.524885, -0.006612, -1.806073, 4.771032
];

% State 3 Parameters (theta_m_dot, parameters 41-60)
theta(41:60) = [
    0.000148, 0.139534, 0.664613, 0.346099, 0.198742, 
    -0.003561, -0.034791, 0.014870, -0.036347, -0.039074, 
    -0.007919, 0.001077, -0.055825, 0.158675, 0.004715, 
    0.095056, -0.005451, -0.003691, 0.044879, -0.837827
];

% State 4 Parameters (theta_L_dot, parameters 61-80)
theta(61:80) = [
    0.000019, 0.206631, 0.642683, 0.184251, 0.080211, 
    -0.014942, -0.153829, 0.009642, 0.068376, -0.042000, 
    -0.007612, -0.014147, -0.031488, -0.176966, 0.002061, 
    -0.049271, -1.343384, 0.001518, 0.419733, -0.803972
];

%% Simulation Setup
t_end = 15;        % simulation time (s)
dt = 0.01;         % time step (s)
N = floor(t_end / dt); % number of iterations
t = 0:dt:t_end;    % time array for plotting

% Original system parameters (for comparison if needed)
Rm = 8.94;         % Motor resistance (Ohm)
Km = 0.0431;       % Motor back-emf constant
Jm = 6e-5;         % Total moment of inertia (kg·m^2)
bm = 3e-4;         % Viscous damping coefficient (Nm/rad/s)
DA = 3e-4;         % Damping coefficient of pendulum arm (Nm/rad/s)
DL = 5e-4;         % Damping coefficient of pendulum link (Nm/rad/s)
mA = 0.053;        % Weight of pendulum arm (kg)
mL = 0.024;        % Weight of pendulum link (kg)
LA = 0.086;        % Length of pendulum arm (m)
LL = 0.128;        % Length of pendulum link (m)
JA = 5.72e-5;      % Inertia moment of pendulum arm (kg·m^2)
JL = 1.31e-4;      % Inertia moment of pendulum link (kg·m^2)
g = 9.81;          % Gravity constant (m/s^2)

% Pre-compute constants for original system model
half_mL_LL_g = 0.5 * mL * LL * g;
half_mL_LL_LA = 0.5 * mL * LL * LA;
quarter_mL_LL_squared = 0.25 * mL * LL^2;

%% Initial Conditions
% Initial state [theta_m, theta_L, theta_m_dot, theta_L_dot]
state_original = [0.5; 0.2; 0; 0];
state_learned = [0.5; 0.2; 0; 0];

% Storage for simulation results
state_original_history = zeros(4, N+1);
state_learned_history = zeros(4, N+1);
vm_history = zeros(1, N);

state_original_history(:, 1) = state_original;
state_learned_history(:, 1) = state_learned;

%% Simulation
for i = 1:N
    % Control input (sinusoidal voltage)
    vm = 5 * sin(t(i));
    vm_history(i) = vm;
    
    %% Simulate Original Physics-Based Model
    % Get current state
    theta_m = state_original(1);
    theta_L = state_original(2);
    theta_m_dot = state_original(3);
    theta_L_dot = state_original(4);
    
    % Motor torque calculation
    im = (vm - Km * theta_m_dot) / Rm;
    Tm = Km * im;
    
    % Equations of motion coefficients
    % For theta_m equation:
    M11 = mL * LA^2 + quarter_mL_LL_squared - quarter_mL_LL_squared * cos(theta_L)^2 + JA;
    M12 = -half_mL_LL_LA * cos(theta_L);
    C1 = 0.5 * mL * LL^2 * sin(theta_L) * cos(theta_L) * theta_m_dot * theta_L_dot;
    C2 = half_mL_LL_LA * sin(theta_L) * theta_L_dot^2;
    
    % For theta_L equation:
    M21 = half_mL_LL_LA * cos(theta_L);
    M22 = JL + quarter_mL_LL_squared;
    C3 = -quarter_mL_LL_squared * cos(theta_L) * sin(theta_L) * theta_m_dot^2;
    G = half_mL_LL_g * sin(theta_L);
    
    % Calculate determinant for matrix inversion
    det_M = M11 * M22 - M12 * M21;
    
    % Handle near-singular matrix
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
    
    % Forward Euler integration for original model
    state_original = state_original + dt * [theta_m_dot; theta_L_dot; theta_m_ddot; theta_L_ddot];
    
    %% Simulate Learned Data-Driven Model
    % Get current state
    y = state_learned;
    
    % Create the regressor vectors
    % First set of regressors (for all states)
    psi1 = zeros(1, 20);
    psi1(1:10) = [1, y(1), y(2), y(3), y(4), ...                   % Constant and linear terms
                  y(1)^2, y(2)^2, y(3)^2, y(4)^2, y(1)*y(2)];      % Squared and cross terms
    psi1(11:20) = [y(1)*y(3), y(1)*y(4), y(2)*y(3), y(2)*y(4), ... % More cross terms
                   y(3)*y(4), sin(y(1)), sin(y(2)), cos(y(1)), ...
                   cos(y(2)), vm];                                  % Trig terms and input
    
    % Second set of regressors (for all states)
    psi2 = zeros(1, 20);
    psi2(1:10) = [y(1)^3, y(2)^3, sin(y(1))*cos(y(2)), ...         % Cubic and combined trig terms 
                  sin(y(2))*cos(y(1)), y(3)*sin(y(2)), ...
                  y(4)*sin(y(2)), vm*y(3), vm^2, ...               % Velocity-angle and voltage terms
                  y(3)*cos(y(2)), y(4)*cos(y(2))];                 
    psi2(11:20) = [y(1)*sin(y(2)), y(2)*sin(y(1)), ...             % Position-angle terms
                   sqrt(abs(y(1))), sqrt(abs(y(2))), ...           % Fractional terms
                   sign(y(3))*y(3)^2, sign(y(4))*y(4)^2, ...       % Nonlinear damping
                   vm*sin(y(2)), vm*y(3)*y(4), ...                 % Mixed terms
                   exp(-abs(y(3))), exp(-abs(y(4)))];              % Exponential terms
    
    % Compute state derivatives using regressors and learned parameters
    state_derivatives = zeros(4, 1);
    
    % State 1 derivative (theta_m_dot)
    state_derivatives(1) = y(3); % Position derivative equals velocity
    
    % State 2 derivative (theta_L_dot)
    state_derivatives(2) = y(4); % Position derivative equals velocity
    
    % Create the full regressor matrix Psi (4x80)
    Psi = zeros(4, 80);
    
    % Position states (theta_m, theta_L)
    Psi(1, 1:20) = psi1;
    Psi(1, 21:40) = psi2;
    Psi(2, 1:20) = psi2;
    Psi(2, 21:40) = psi1;
    
    % Velocity states (theta_m_dot, theta_L_dot)
    Psi(3, 41:60) = psi1;
    Psi(3, 61:80) = psi2;
    Psi(4, 41:60) = psi2;
    Psi(4, 61:80) = psi1;
    
    % Compute state derivatives using the regressor matrix and parameter vector
    state_derivatives = zeros(4, 1);
    
    % Position derivatives (equal to velocity states)
    state_derivatives(1) = y(3);
    state_derivatives(2) = y(4);
    
    % Acceleration derivatives (computed from parameters)
    state_derivatives(3) = Psi(3,:) * theta;
    state_derivatives(4) = Psi(4,:) * theta;
    
    % Forward Euler integration for learned model
    state_learned = state_learned + dt * state_derivatives;
    
    % Normalize the link angle to [-π, π] for both models
    state_original(2) = mod(state_original(2), 2*pi);
    if state_original(2) > pi
        state_original(2) = state_original(2) - 2*pi;
    end
    
    state_learned(2) = mod(state_learned(2), 2*pi);
    if state_learned(2) > pi
        state_learned(2) = state_learned(2) - 2*pi;
    end
    
    % Store results
    state_original_history(:, i+1) = state_original;
    state_learned_history(:, i+1) = state_learned;
end

%% Plotting Results

% Figure 1: Position states comparison
figure(1)
clf;
subplot(2,1,1)
plot(t, state_original_history(1,:), 'b-', 'LineWidth', 2)
hold on
plot(t, state_learned_history(1,:), 'r--', 'LineWidth', 2)
legend('Original Model', 'Learned Model')
grid on
grid minor
ylabel('\theta_m [rad]')
xlim([0 t_end])
ylim([-20 0])
title('Comparison of Original vs. Learned Model: Position States')

subplot(2,1,2)
plot(t, state_original_history(2,:), 'b-', 'LineWidth', 2)
hold on
plot(t, state_learned_history(2,:), 'r--', 'LineWidth', 2)
grid on
grid minor
xlim([0 t_end])
ylabel('\theta_L [rad]')
xlabel('time (s)')

% Figure 2: Velocity states comparison
figure(2)
clf;
subplot(2,1,1)
plot(t, state_original_history(3,:), 'b-', 'LineWidth', 2)
hold on
plot(t, state_learned_history(3,:), 'r--', 'LineWidth', 2)
legend('Original Model', 'Learned Model')
grid on
grid minor
ylabel('\theta_m dot [rad/s]')
xlim([0 t_end])
title('Comparison of Original vs. Learned Model: Velocity States')

subplot(2,1,2)
plot(t, state_original_history(4,:), 'b-', 'LineWidth', 2)
hold on
plot(t, state_learned_history(4,:), 'r--', 'LineWidth', 2)
grid on
grid minor
xlim([0 t_end])
ylabel('\theta_L dot [rad/s]')
xlabel('time (s)')

% Figure 3: Model Error
figure(3)
clf;
subplot(4,1,1)
error_theta_m = state_original_history(1,:) - state_learned_history(1,:);
plot(t, error_theta_m, 'k-', 'LineWidth', 2)
grid on
grid minor
ylabel('Error \theta_m [rad]')
xlim([0 t_end])
title('Model Error (Original - Learned)')

subplot(4,1,2)
error_theta_L = state_original_history(2,:) - state_learned_history(2,:);
plot(t, error_theta_L, 'k-', 'LineWidth', 2)
grid on
grid minor
ylabel('Error \theta_L [rad]')
xlim([0 t_end])

subplot(4,1,3)
error_theta_m_dot = state_original_history(3,:) - state_learned_history(3,:);
plot(t, error_theta_m_dot, 'k-', 'LineWidth', 2)
grid on
grid minor
ylabel('Error \theta_m dot [rad/s]')
xlim([0 t_end])

subplot(4,1,4)
error_theta_L_dot = state_original_history(4,:) - state_learned_history(4,:);
plot(t, error_theta_L_dot, 'k-', 'LineWidth', 2)
grid on
grid minor
ylabel('Error \theta_L dot [rad/s]')
xlabel('time (s)')
xlim([0 t_end])

% Figure 4: Input voltage
figure(4)
clf;
plot(t(1:end-1), vm_history, 'LineWidth', 2)
grid on
grid minor
title('Control Input (Motor Voltage)')
ylabel('Voltage (V)')
xlabel('Time (s)')
xlim([0 t_end])

% Calculate RMSE between original and learned models
rmse_theta_m = rms(error_theta_m);
rmse_theta_L = rms(error_theta_L);
rmse_theta_m_dot = rms(error_theta_m_dot);
rmse_theta_L_dot = rms(error_theta_L_dot);

fprintf('\n===== MODEL COMPARISON RESULTS =====\n');
fprintf('RMSE for theta_m: %.6f rad\n', rmse_theta_m);
fprintf('RMSE for theta_L: %.6f rad\n', rmse_theta_L);
fprintf('RMSE for theta_m_dot: %.6f rad/s\n', rmse_theta_m_dot);
fprintf('RMSE for theta_L_dot: %.6f rad/s\n', rmse_theta_L_dot);
fprintf('Total RMSE: %.6f\n', rmse_theta_m + rmse_theta_L + rmse_theta_m_dot + rmse_theta_L_dot);

% Calculate normalized RMSE
range_theta_m = max(state_original_history(1,:)) - min(state_original_history(1,:));
range_theta_L = max(state_original_history(2,:)) - min(state_original_history(2,:));
range_theta_m_dot = max(state_original_history(3,:)) - min(state_original_history(3,:));
range_theta_L_dot = max(state_original_history(4,:)) - min(state_original_history(4,:));

nrmse_theta_m = rmse_theta_m / range_theta_m * 100;
nrmse_theta_L = rmse_theta_L / range_theta_L * 100;
nrmse_theta_m_dot = rmse_theta_m_dot / range_theta_m_dot * 100;
nrmse_theta_L_dot = rmse_theta_L_dot / range_theta_L_dot * 100;

fprintf('\n===== NORMALIZED RMSE (%) =====\n');
fprintf('NRMSE for theta_m: %.2f%%\n', nrmse_theta_m);
fprintf('NRMSE for theta_L: %.2f%%\n', nrmse_theta_L);
fprintf('NRMSE for theta_m_dot: %.2f%%\n', nrmse_theta_m_dot);
fprintf('NRMSE for theta_L_dot: %.2f%%\n', nrmse_theta_L_dot);
fprintf('Average NRMSE: %.2f%%\n', (nrmse_theta_m + nrmse_theta_L + nrmse_theta_m_dot + nrmse_theta_L_dot)/4);
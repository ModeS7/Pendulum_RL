%% Simple Simulation of phi and theta dynamics
clear;
clc;

%% Simulation Parameters
max_voltage = 5.0;      % Maximum voltage [V]
dt = 0.001;             % Time step [s]
tf = 10;                % Total simulation time [s]
t = 0:dt:tf;            % Time vector

%% System Parameters
g   = 9.81;                 % Gravitational acceleration [m/s^2]
m_p = 0.024;                % Mass [kg]
l   = 0.129 / 2.0;          % Effective length [m]
J_p = m_p * (l * 2)^2 / 3;  % Moment of inertia

%% Initial Conditions
% For phi (translational dynamics)
phi       = 0;        % Initial phi position
phidot    = 0;        % Initial phi velocity
phidotdot = 0;        % Initial phi acceleration

% For theta (rotational dynamics)
th       = 2;         % Initial theta angle (in radians)
thdot    = 0;         % Initial theta angular velocity
thdotdot = 0;         % Initial theta angular acceleration

%% Preallocate Arrays for Storing Simulation Data
phi_array    = zeros(1, length(t));
phidot_array = zeros(1, length(t));
th_array     = zeros(1, length(t));
thdot_array  = zeros(1, length(t));

%% Control Input
% Define a constant control input 'u' (ensure it does not exceed max_voltage)
u = 1.0;  % Example: you can change this value or implement a control strategy

%% Simulation Loop
for i = 1:length(t)


    % --- Update phi dynamics ---
    % new_phidotdot = phidotdot + (u/19)*dt
    new_phidotdot = (u / 19);
    new_phidot    = phidot + new_phidotdot * dt;
    new_phi       = phi + new_phidot * dt;
    
    % --- Update theta dynamics ---
    % new_thdotdot = thdotdot + {[-m_p * g * l * 0.5 * sin(th) + m_p * l * 0.5 * (u/1.615) * cos(th)]/J_p}*dt
    new_thdotdot = -((m_p * g * l * 0.5 * sin(th)) + (m_p * l * 0.5 * (u / 1.615) * cos(th))) / J_p;
    new_thdot    = thdot + new_thdotdot * dt;
    new_th       = th + new_thdot * dt;
    
    % Store current state values for plotting
    phi_array(i)    = phi;
    phidot_array(i) = phidot;
    th_array(i)     = th;
    thdot_array(i)  = thdot;
    
    % Update states for the next time step
    phi       = new_phi;
    phidot    = new_phidot;
    phidotdot = new_phidotdot;
    
    th        = new_th;
    thdot     = new_thdot;
    thdotdot  = new_thdotdot;
end

%% Plot the Results
figure(1);
clf;
subplot(2,1,1);
plot(t, phi_array, 'b-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('\phi');
title('Evolution of \phi over Time');
grid on;

subplot(2,1,2);
plot(t, th_array, 'r-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('\theta (rad)');
title('Evolution of \theta over Time');
grid on;

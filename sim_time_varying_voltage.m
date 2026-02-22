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
th       = 0;         % Initial theta angle (in radians)
thdot    = 0;         % Initial theta angular velocity
thdotdot = 0;         % Initial theta angular acceleration

%% Preallocate Arrays for Storing Simulation Data
phi_array    = zeros(1, length(t));
phidot_array = zeros(1, length(t));
th_array     = zeros(1, length(t));
thdot_array  = zeros(1, length(t));

%% Control Input
% Define a constant control input 'u' (ensure it does not exceed max_voltage)
u = 0;  % Example: you can change this value or implement a control strategy

%% Simulation Loop
for i = 1:length(t)

    if i>500
        u = -5;
    end
    if i>1500
        u = 0.1;
    end
    if i>2500
        u = 0.1;
    end
    if i>3500
        u = 0.1;
    end

    new_phi = phi + phidot *dt;
    new_th = th + thdot * dt;

    %new_phidot = phidot + (m_p^2 * L_p^2 * L_r * g / 4 -(J_p + m_p * L_p^2 / 4) * D_r + m_p * L_p * L_r * D_p) * dt;
    
    new_phidot = phidot + (149.3 * th - 14.93 * phidot + 4.915 * thdot + 49.73 * u) * dt;
    new_thdot = thdot + (-261.6 * th + 14.76 * phidot - 8.614 * thdot - 49.15 * u) * dt;
    
    %new_phidot = phidot + (-41.6 * th - 4.16 * phidot + 1.37 * thdot + 13.9 * u) * dt;
    %new_thdot = thdot + (72.4 * th - 4.11 * phidot - 2.4 * thdot + 13.7 * u) * dt;
    
    %new_phidot = phidot + (u / 19) * dt;
    %new_thdot = thdot + -((m_p * g * l * 0.5 * sin(th)) + (m_p * l * 0.5 * (u / 1.615) * cos(th))) / J_p * dt;

    % Store current state values for plotting
    phi_array(i)    = phi;
    phidot_array(i) = phidot;
    th_array(i)     = th;
    thdot_array(i)  = thdot;
    
    % Update states for the next time step
    phi       = new_phi;
    phidot    = new_phidot;
    
    th        = new_th;
    thdot     = new_thdot;
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

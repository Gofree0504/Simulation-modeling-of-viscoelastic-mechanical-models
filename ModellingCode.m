% =========================================================================
% # Tumor tissue viscoelasticity extremum search model                    #
% # ModelingCode.m - Finally revised on Mar 2026                          #
% # Coded by Ruan Yueheng                                                 #
% # Copyright (C) Ruan Yueheng                                            #
% #########################################################################
% =========================================================================

clear; clc; close all;

%% ========================================================================
%  1. Experimental loading parameter settings
% ========================================================================
eps_max = 0.10;    
t_ramp = 0.165;    
t_total = 120;     
t_span = linspace(0, t_total, 500); 

% Set the stretching rate function
strain_func = @(t) min(eps_max, (t/t_ramp)*eps_max);
strain_rate_func = @(t) (t < t_ramp) * (eps_max / t_ramp);

%% ========================================================================
%  2. Data import and denoising enhancement
% ========================================================================
data_folder = 'C:\Users\Yourname\Desktop\Simulation Modeling\Dataset\InputData\'; 
file_list = dir(fullfile(data_folder, 'sample_tau*_data.xlsx'));
num_samples = length(file_list);

if num_samples == 0
    error('No data file was found. Please check the folder path!');
end

exp_data = struct();
ydata_all = []; 

smooth_span = 0.10; 

for k = 1:num_samples
    file_name = file_list(k).name;
    tokens = regexp(file_name, 'tau([\d\.]+)_', 'tokens');
    exp_data(k).tau = str2double(tokens{1}{1});
    
    curr_data = readmatrix(fullfile(data_folder, file_name));
    
    % --- Data Sorting ---
    [raw_t, sort_idx] = sort(curr_data(:, 1));
    raw_stress = curr_data(sort_idx, 2);
    
    % --- Robust Smoothing Denoising ---
    smooth_stress = smooth(raw_t, raw_stress, smooth_span, 'rloess');
    
    % --- Avoiding non-physical negative stresses ---
    smooth_stress = max(smooth_stress, 0); 

    exp_data(k).t = raw_t;
    exp_data(k).stress_raw = raw_stress;       
    exp_data(k).stress_smooth = smooth_stress; 
    
    ydata_all = [ydata_all; smooth_stress]; 
end

fprintf('Successfully loaded and removed anomalous noise from %d samples, ready to initiate ODE optimization.\n', num_samples);

%% ========================================================================
%  3. Global optimal scaling law solving
% ========================================================================
% P = [ E_eq_base, E1_base, E2_base, lam1_base, lam2_base, alpha, gamma ]

initial_guess = [200, 15000, 3000, 0.25, 50.0, 1.0, -1.0]; 

lb = [  0,  1000,   100, 0.01,  1.0,  0.0, -5.0]; 
ub = [1e4,  1e6,    1e5,  5.0,  500,  5.0,  0.0]; 

xdata_dummy = (1:num_samples)'; 
lsq_target_func = @(P, ~) objective_ode(P, xdata_dummy, exp_data, t_span, strain_func, strain_rate_func);

options = optimoptions('lsqcurvefit', ...
    'Display', 'iter', 'MaxIterations', 50, 'StepTolerance', 1e-4);

disp('Initiating full spatiotemporal optimization based on underlying ODEs using denoised data. Please wait...');
[P_opt, resnorm, residual] = lsqcurvefit(lsq_target_func, initial_guess, xdata_dummy, ydata_all, lb, ub, options);

%% ========================================================================
%  4. Core metric calculation only (Non-plotting version)
% ========================================================================
% Calculate global goodness-of-fit R^2 (Essential metric for publication)
SS_tot_global = sum((ydata_all - mean(ydata_all)).^2);
R2_global = 1 - (sum(residual.^2) / SS_tot_global);
fprintf('Global goodness-of-fit of the model R^2 = %.4f\n', R2_global);

% If you need to save the fitted curve data, keep the calculation logic inside the loop.
% If exporting to other software is not needed, you can delete the plot/scatter commands hereafter.

%% ========================================================================
%  5. Plot core mechanism: Effective relaxation time vs. Tortuosity
% ========================================================================
tau_values_plot = [exp_data.tau];
lam_eff_list = zeros(1, num_samples);

% --- Keep original physical calculation workflow ---
for k = 1:num_samples
    tau = tau_values_plot(k);
    E1_curr = P_opt(2) * tau^P_opt(6);
    E2_curr = P_opt(3) * tau^P_opt(6);
    lam1_curr = P_opt(4) * tau^P_opt(7);
    lam2_curr = P_opt(5) * tau^P_opt(7);
    % Calculate effective relaxation time (Original logic: modulus-weighted average)
    lam_eff_list(k) = (E1_curr*lam1_curr + E2_curr*lam2_curr) / (E1_curr + E2_curr);
end

figure('Position', [950, 200, 500, 450], 'Color', 'w');
hold on;

% 1. Define custom colors (Low tortuosity Blue #5c96d0 -> High tortuosity Pink #ee7c7c)
color_blue = [92, 150, 208] / 255; 
color_pink = [238, 124, 124] / 255; 

% 2. Plot linear trend line (Deep dark gray, bold width 2.5)
poly_p = polyfit(tau_values_plot, lam_eff_list, 1);
trend_x = linspace(min(tau_values_plot)*0.95, max(tau_values_plot)*1.05, 50);
trend_y = polyval(poly_p, trend_x);
plot(trend_x, trend_y, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 2.5, ...
     'DisplayName', 'Linear fitting trend');

% 3. Plot scatter points (Colors matching normalized background, 20% opacity)
% Manually assign color to each point to control opacity
for i = 1:num_samples
    % Calculate the ratio of current tau in the color band
    ratio = (tau_values_plot(i) - min(tau_values_plot)) / (max(tau_values_plot) - min(tau_values_plot));
    current_color = (1-ratio)*color_blue + ratio*color_pink;
    
    scatter(tau_values_plot(i), lam_eff_list(i), 120, current_color, 'o', 'filled', ...
            'MarkerFaceAlpha', 0.20, ...      % Background fill with 20% opacity
            'MarkerEdgeColor', current_color, ... % Solid edge to ensure clarity
            'LineWidth', 1.5, 'HandleVisibility', 'off');
end

% 4. In-depth style embellishment
xlabel('Network Tortuosity \tau', 'FontSize', 12, 'FontWeight', 'bold'); 
ylabel('Effective Relaxation Time \lambda_{eff} (s)', 'FontSize', 12, 'FontWeight', 'bold');
title('Core Finding: Tortuosity Enhances Viscoelastic Longevity', 'FontSize', 13);

grid on; 
box off; % Use open box style (turn off top and right borders) for a cleaner look
set(gca, 'FontSize', 11, 'LineWidth', 1.1, 'TickDir', 'out');

% 5. Configure right colorbar (displaying tortuosity values)
custom_map = [linspace(color_blue(1), color_pink(1), 256)', ...
              linspace(color_blue(2), color_pink(2), 256)', ...
              linspace(color_blue(3), color_pink(3), 256)'];
colormap(gca, custom_map);
c = colorbar;
c.Label.String = 'Tortuosity Value \tau';
c.Label.FontWeight = 'bold';
caxis([min(tau_values_plot), max(tau_values_plot)]); % Ensure colorbar ticks correspond to tortuosity values

% Adjust display range
xlim([min(tau_values_plot)*0.95, max(tau_values_plot)*1.05]);
ylim([min(lam_eff_list)*0.9, max(lam_eff_list)*1.1]);

%% ========================================================================
% [Local Object Function] LSQ ODE black-box calculator
% ========================================================================
function Y_pred_all = objective_ode(P, xdata_idx, exp_data, t_span, strain_func, strain_rate_func)
    Y_pred_all = [];
    for i = 1:length(xdata_idx)
        k = xdata_idx(i);
        tau = exp_data(k).tau;
        t_exp = exp_data(k).t; 
        
        E_eq = P(1) * tau^P(6);
        E1   = P(2) * tau^P(6);
        E2   = P(3) * tau^P(6);
        lam1 = P(4) * tau^P(7);
        lam2 = P(5) * tau^P(7);
        
        ode_sys = @(t,y) [ E1*strain_rate_func(t) - y(1)/lam1; 
                           E2*strain_rate_func(t) - y(2)/lam2 ];
        [~, Y_sol] = ode45(ode_sys, t_span, [0; 0]);
        
        sim_elastic = arrayfun(strain_func, t_span)' .* E_eq;
        total_stress = sim_elastic + Y_sol(:,1) + Y_sol(:,2);
        
        % Interpolation for data points
        sim_at_exp = interp1(t_span, total_stress, t_exp, 'linear', 'extrap');
        Y_pred_all = [Y_pred_all; sim_at_exp]; 
    end
end

%% ========================================================================
%  Final results output: Normalized relaxation curves (Custom colors)
% ========================================================================
figure('Position', [150, 150, 850, 600], 'Color', 'w');
hold on;

% 1. Prepare sorting (Sort by tau ascendingly for color mapping)
tau_all = [exp_data.tau];
[sorted_tau, sort_idx] = sort(tau_all);

% 2. Define specified exact colors (Convert to RGB 0-1 range)
% Low tortuosity end: #5c96d0 (Blue)
color_blue = [92, 150, 208] / 255; 
% High tortuosity end: #ee7c7c (Pink)
color_pink = [238, 124, 124] / 255; 

% Generate interpolated color band from Blue (Low tau) to Pink (High tau)
colors_custom = [linspace(color_blue(1), color_pink(1), num_samples)', ...
                 linspace(color_blue(2), color_pink(2), num_samples)', ...
                 linspace(color_blue(3), color_pink(3), num_samples)'];

norm_res_all = []; 
norm_total_all = [];

for i = 1:num_samples
    k = sort_idx(i);
    tau = exp_data(k).tau;
    
    % --- Data truncation: Strictly starting from 0.143s ---
    valid_idx = exp_data(k).t >= 0.143;
    t_exp = exp_data(k).t(valid_idx);
    stress_smooth = exp_data(k).stress_smooth(valid_idx);
    
    % --- Normalization of experimental data ---
    max_S = max(stress_smooth);
    norm_S_exp = stress_smooth / max_S;
    
    % --- Generate high-resolution model predictions ---
    t_dense = logspace(log10(0.143), log10(max(t_exp)), 800); 
    
    % Extract optimal parameters (Ensure P_opt is solved above)
    E_eq = P_opt(1) * tau^P_opt(6);
    E1   = P_opt(2) * tau^P_opt(6);
    E2   = P_opt(3) * tau^P_opt(6);
    lam1 = P_opt(4) * tau^P_opt(7);
    lam2 = P_opt(5) * tau^P_opt(7);
    
    % Solve ODEs
    ode_sys = @(t,y) [ E1*strain_rate_func(t) - y(1)/lam1; 
                       E2*strain_rate_func(t) - y(2)/lam2 ];
    [~, Y_sol] = ode45(ode_sys, t_dense, [0; 0]);
    sim_elastic = arrayfun(strain_func, t_dense) .* E_eq;
    sim_S_dense = sim_elastic(:) + Y_sol(:,1) + Y_sol(:,2);
    norm_S_sim_dense = sim_S_dense / max(sim_S_dense);
    
    % Calculate R^2 for current sample
    [~, Y_res] = ode45(ode_sys, t_exp, [0; 0]);
    sim_S_res = arrayfun(strain_func, t_exp) .* E_eq + Y_res(:,1) + Y_res(:,2);
    norm_S_sim_res = sim_S_res / max(sim_S_res);
    norm_res_all = [norm_res_all; (norm_S_exp - norm_S_sim_res)];
    norm_total_all = [norm_total_all; (norm_S_exp - mean(norm_S_exp))];

    % --- Plotting ---
    % Background raw data: Apply 20% opacity (MarkerFaceAlpha = 0.2)
    scatter(t_exp, norm_S_exp, 30, colors_custom(i,:), 'o', 'filled', ...
            'MarkerFaceAlpha', 0.2, 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Model fitted solid line: Bright and solid colors
    plot(t_dense, norm_S_sim_dense, '-', 'Color', colors_custom(i,:), 'LineWidth', 2.5, ...
         'DisplayName', ['\tau = ', num2str(tau, '%.2f')]);
end

% Calculate global normalized R^2
mask = ~isnan(norm_res_all);
R2_norm = 1 - (sum(norm_res_all(mask).^2) / sum(norm_total_all(mask).^2));

% --- Style Enhancement ---
set(gca, 'XScale', 'log', 'FontSize', 11, 'LineWidth', 1.1, 'TickDir', 'out', 'Box', 'off'); 
xlim([0.143, max(t_exp)]); 
ylim([0, 1.05]);

xlabel('Testing Time t (s) [Log Scale]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized Stress \sigma / \sigma_{max}', 'FontSize', 12, 'FontWeight', 'bold');
title(['Normalized Kinetic Response (Model R^2 = ', num2str(R2_norm, '%.3f'), ')'], 'FontSize', 14);

% Plot custom legend and colorbar
colormap(gca, colors_custom);
c = colorbar;
c.Label.String = 'Tortuosity Gradient (\tau: Blue \rightarrow Pink)';
c.Label.FontSize = 11;
caxis([min(tau_all), max(tau_all)]);

grid on;
ax = gca;
ax.GridAlpha = 0.3; % Lighten grid for a cleaner appearance
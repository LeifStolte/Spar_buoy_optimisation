function [h_opt, varargout] = check_freq_gradients(x)
    % ====================================================================
    % == TASK 5: SELF-CONTAINED ONE-TIME STEP SIZE OPTIMIZATION SWEEP   ==
    % ====================================================================
    
    %% 1. COMPUTE BASELINE CONSTRAINTS & ANALYTICAL GRADIENTS
    % Evaluates the initial state to populate c_base and the Jacobian gc_base
    [c_base, ~, gc_base, ~] = compute_constraints(x);

    fprintf('\n=============================================================\n');
    fprintf('   RUNNING ONE-TIME FINITE DIFFERENCE STEP SIZE SWEEP       \n');
    fprintf('=============================================================\n');
    
    % Sweep down across the full floating-point envelope (10^1 to 10^-10)
    h_vect = logspace(1, -10, 20);
    
    % Storage arrays for relative error lines
    errors_D = zeros(size(h_vect));
    errors_t = zeros(size(h_vect));
    
    % Element index and constraint index to test against
    ee_test = 1;                         % Element 1
    idx_con = length(c_base);            % Final constraint entry (1st Eigenfrequency)
    
    % Determine total design variables safely (Half are D, half are t)
    num_vars = length(x);
    num_sec = num_vars / 2;              % Safely computes number of sections dynamically
    
    % Identify variable indices inside the flat 'x' vector layout
    idx_D = ee_test;                     
    idx_t = num_sec + ee_test; 
    
    % Extract analytical baselines from the complete constraint Jacobian matrix
    analytical_grad_D = gc_base(idx_D, idx_con);
    analytical_grad_t = gc_base(idx_t, idx_con);
    
    fprintf('Analytical Baselines - Diameter (D): %e | Thickness (t): %e\n\n', analytical_grad_D, analytical_grad_t);
    fprintf('%-15s | %-20s | %-20s\n', 'Step Size (h)', 'Rel Error (Diameter)', 'Rel Error (Thickness)');
    fprintf('-----------------------------------------------------------------------------\n');
    
    %% 2. FINITE DIFFERENCE EVALUATION LOOP
    for i = 1:length(h_vect)
        h = h_vect(i);
        
        % --- 1. Forward Difference for Diameter (D) ---
        x_perturbed_D = x;
        x_perturbed_D(idx_D) = x(idx_D) + h;
        [c_perturbed_D, ~, ~, ~] = compute_constraints(x_perturbed_D);
        fd_grad_D = (c_perturbed_D(idx_con) - c_base(idx_con)) / h;
        
        if abs(analytical_grad_D) > 1e-9
            errors_D(i) = abs(fd_grad_D - analytical_grad_D) / abs(analytical_grad_D);
        else
            errors_D(i) = abs(fd_grad_D - analytical_grad_D);
        end
        
        % --- 2. Forward Difference for Thickness (t) ---
        x_perturbed_t = x;
        x_perturbed_t(idx_t) = x(idx_t) + h;
        [c_perturbed_t, ~, ~, ~] = compute_constraints(x_perturbed_t);
        fd_grad_t = (c_perturbed_t(idx_con) - c_base(idx_con)) / h;
        
        if abs(analytical_grad_t) > 1e-9
            errors_t(i) = abs(fd_grad_t - analytical_grad_t) / abs(analytical_grad_t);
        else
            errors_t(i) = abs(fd_grad_t - analytical_grad_t);
        end
        
        fprintf('%-15e | %-20e | %-20e\n', h, errors_D(i), errors_t(i));
    end
    
    [min_err_D, idx_opt_D] = min(errors_D);
    [min_err_t, idx_opt_t] = min(errors_t);
    
    fprintf('-----------------------------------------------------------------------------\n');
    fprintf('Optimal h for Diameter : %e (Min Error: %e)\n', h_vect(idx_opt_D), min_err_D);
    fprintf('Optimal h for Thickness: %e (Min Error: %e)\n', h_vect(idx_opt_t), min_err_t);
    fprintf('=============================================================\n\n');
    
    %% 3. ASSIGN FUNCTION OUTPUT VALUES
    % Since thickness features smaller absolute scales, its round-off limit 
    % is usually more sensitive. We pass it as the primary scalar h_opt.
    h_opt = h_vect(idx_opt_t); 
    
    if nargout > 1
        varargout{1} = h_vect(idx_opt_D);
    end

    %% 4. LATEX HALF-PAGE FORMATTING PIPELINE (0.5 \textwidth sizing)
    fig = figure('Name', 'Gradient Verification: Variable Parameter Comparison', ...
                 'Units', 'centimeters');
    
    % Standard half-width text block layout bounds on an A4 document (~8.5 cm width)
   
    
    % Render the V-Curves
    hD = loglog(h_vect, errors_D, '-ro', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'r'); hold on;
    ht = loglog(h_vect, errors_t, '-bs', 'LineWidth', 1.5, 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    
    % Grid settings
    grid on;
    set(gca, 'XDir', 'normal', 'GridLineStyle', ':', 'GridAlpha', 0.4);
    
    % Set fonts and physical axis padding bounds cleanly
    set(gca, 'FontName', 'Helvetica', 'FontSize', 22, 'LineWidth', 1.0);
    
    % Label adjustments 
    xlabel('Step Size, $h$', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('Relative Error, $\epsilon_{\mathrm{rel}}$', 'Interpreter', 'latex', 'FontSize', 18);
    
    % Compact Legend configuration
    legend([hD, ht], {'Diameter ($D$)', 'Thickness ($t$)'}, ...
                 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 16);
    legend('boxoff');
    
    % Enforce precise visual boundaries
    axis tight;
    

    hold off;
    
    drawnow;
end
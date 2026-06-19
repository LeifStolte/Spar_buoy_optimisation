function [ w, dw ] = compute_frequency( problem, beam, solution )
    %% 1. AUXILIARY CONFIGURATIONS
    N = problem.info.num_sec;
    ifreq = 1:length(problem.constraints.max_freq);
    
    % High-Efficiency Pass: Transpose edof once outside the loops
    % This saves millions of redundant matrix operations during fmincon runs
    if size(problem.frans.edof_1d, 1) == N
        edof_matrix = problem.frans.edof_1d;
    else
        edof_matrix = problem.frans.edof_1d'; 
    end
    
    %% 2. FREQUENCY & GRADIENT INITIALIZATION
    w = solution.eigfreq(ifreq);    % w = omega^2 in [ (rad/s)^2 ]
    vec = solution.eigvec(:,ifreq); % Mass-normalized eigenvector matrix
    
    dw.grad = cell(problem.nvar, 1);
    for grd = 1:problem.nvar
        dw.grad{grd} = zeros(N, length(ifreq));
    end
    
    %% 3. EIGENFREQUENCY GRADIENT CALCULATION LOOP
    for iq = 1:length(ifreq)
        lambda = w(iq);
        
        % Extract the specific mode vector column for the current frequency
        phi = vec(:, iq); 
        
        for ie = 1:N
            % Extract local element degrees of freedom
            ed = edof_matrix(ie, :);
            ve = phi(ed); % Localized displacement vector for this element
            
            for grd = 1:problem.nvar
                % Pull structural matrix derivatives
                dKe = beam.grad{grd}.Ke(:,:,ie);
                dMe = beam.grad{grd}.Me(:,:,ie);
                
                % Compute core structural gradient contribution
                dw.grad{grd}(ie, iq) = ve' * (dKe - lambda * dMe) * ve;
                
                % Apply soil stiffness modification for monopiles
                if strcmp(problem.type, 'tower_monopile')
                    if grd == 1 && ie <= problem.info.mudline_elms
                        if isfield(beam, 'dKsoil')
                            
                            % Slice the 3D array to isolate the current element's soil properties
                            dKe_soil = beam.dKsoil(:,:,ie);
                            
                            % Dimension Guard: Dynamically handle local vs global matrix layouts
                            if size(dKe_soil, 1) == length(ed)
                                % Local layout (e.g. 12x12) -> multiply by local element vector 've'
                                dw.grad{grd}(ie, iq) = dw.grad{grd}(ie, iq) + ve' * dKe_soil * ve;
                            else
                                % Global layout -> multiply by the specific single mode shape vector 'phi'
                                dw.grad{grd}(ie, iq) = dw.grad{grd}(ie, iq) + phi' * dKe_soil * phi;
                            end
                            
                        end
                    end
                end % ends tower_monopile checks
                
            end % ends grd loop
        end % ends ie loop
    end % ends iq loop
end
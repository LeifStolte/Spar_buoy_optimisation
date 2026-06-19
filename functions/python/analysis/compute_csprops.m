function [ csprops ] = compute_csprops( problem, design )
% Analyze the cross section properties

	if strcmp(problem.type,'blade_structure')

		%Caps
		[ csprops.caps ] = compute_caps_csprops( problem, design );

		%Ellipse
		[ csprops.ellipse] = compute_ellipse_csprops( problem, design );

		%Caps + ellipse
		csprops.total=csprops.caps;
		csprops.total = returnStruct(csprops.total, csprops.ellipse);

	elseif strcmp(problem.type, 'tower_monopile')

		csprops.tower_monopile = compute_tower_monopile_csprops( problem, design );
		csprops.total = csprops.tower_monopile;

	else

		disp('problem type not implemented');
		error();

	end

    function rs = returnStruct(s,a)
        fn = fieldnames(s);
        for i=1:length(fn)
            if isstruct(s.(fn{i}))
                s.(fn{i}) = returnStruct(s.(fn{i}), a.(fn{i}));
			elseif iscell(s.(fn{i}))
				N_cell=size(s.(fn{i}),2);
				for j=1:N_cell
					s.(fn{i}){j} = returnStruct(s.(fn{i}){j}, a.(fn{i}){j});
				end
            else
				% disp('----');
				% disp(fn{i});
                s.(fn{i}) = s.(fn{i})+a.(fn{i});
            end
        end
        rs = s;
    end

end


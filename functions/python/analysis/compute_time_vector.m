function [ times ] = compute_time_vector(t0, tf, dt)
	% initialize some of the data structures
	times=[t0];
	t=t0;
	while t < tf
		t=t+dt;
		times(end+1)=t;
	end
end



function [ Mcm ] = compute_tower_monopile_concentrated_mass_matrix( problem )

    % extract all the parameters
	N=problem.info.num_sec;
	Ntp=problem.info.transition_piece_elms;
	Mtp=problem.info.transition_piece_mass;
	Mrna=problem.info.hub_nacelle_mass+problem.info.rotor_mass;
	Irna_xz=problem.info.rotor_inertia_x_z;
	Irna_y=problem.info.rotor_inertia_y;

	% allocate our matrix
	Ndof=(1+N*3)*6;
	Mcm=zeros(Ndof);

	% add the TP mass
	DofTp=Ntp*18;
	Mcm(DofTp+1,DofTp+1)=Mtp;
	Mcm(DofTp+2,DofTp+2)=Mtp;
	Mcm(DofTp+3,DofTp+3)=Mtp;

	% add the RNA mass and Inertia
	DofRNA=N*18;
	Mcm(DofRNA+1,DofRNA+1)=Mrna;
	Mcm(DofRNA+2,DofRNA+2)=Mrna;
	Mcm(DofRNA+3,DofRNA+3)=Mrna;
	Mcm(DofRNA+4,DofRNA+4)=Irna_xz;
	Mcm(DofRNA+5,DofRNA+5)=Irna_y;
	Mcm(DofRNA+6,DofRNA+6)=Irna_xz;

end

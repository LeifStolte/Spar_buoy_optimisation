function [ Ksoil, dKsoil ] = compute_soil_stiffness_matrix( problem, design )

	% Extract params and Initialize the data
	N=problem.info.num_sec;
	ndof=(1+3*N)*6;
	soil_stiff=problem.info.soil_coeff_subgrade_reaction;
	Ksoil=zeros(ndof);
	dKsoil=zeros(ndof,ndof,problem.info.mudline_elms);

	% Initialize the gradient scalar
	dK=soil_stiff*problem.info.delta_z*0.5;

	% calculate the soil stiffness
	for i=1:problem.info.mudline_elms
		dof_low=1+(i-1)*18;
		dof_hgh=19+(i-1)*18;
		% problem.info.mudline_elms
		K_per_node=dK*design.D(i);
		Ksoil(dof_low  ,dof_low  )=Ksoil(dof_low  ,dof_low  )+K_per_node;
		Ksoil(dof_low+1,dof_low+1)=Ksoil(dof_low+1,dof_low+1)+K_per_node;
		Ksoil(dof_hgh  ,dof_hgh  )=Ksoil(dof_hgh  ,dof_hgh  )+K_per_node;
		Ksoil(dof_hgh+1,dof_hgh+1)=Ksoil(dof_hgh+1,dof_hgh+1)+K_per_node;

		dKsoil(dof_low  ,dof_low  ,i)=dKsoil(dof_low  ,dof_low  ,i)+dK;
		dKsoil(dof_low+1,dof_low+1,i)=dKsoil(dof_low+1,dof_low+1,i)+dK;
		dKsoil(dof_hgh  ,dof_hgh  ,i)=dKsoil(dof_hgh  ,dof_hgh  ,i)+dK;
		dKsoil(dof_hgh+1,dof_hgh+1,i)=dKsoil(dof_hgh+1,dof_hgh+1,i)+dK;
	end

	%for i=1:ndof
	%  for j=1:ndof
	%	  Ksoil_val=Ksoil(i,j);
	%	  dKsoil_val_1=dKsoil(i,j,1);
	%	  dKsoil_val_2=dKsoil(i,j,2);
	%	  dKsoil_val_3=dKsoil(i,j,3);
	%	  dKsoil_val_4=dKsoil(i,j,4);
	%	  if Ksoil_val~=0.0 | dKsoil_val_1~=0.0 | dKsoil_val_2~=0.0 | dKsoil_val_3~=0.0 | dKsoil_val_3~=0.0
	%		  fprintf('@(%i,%i) K: %e dK1: %e dK2: %e dK3: %e dK4: %e\n', i, j, Ksoil_val, dKsoil_val_1, dKsoil_val_2, dKsoil_val_3, dKsoil_val_4);
	%	  end
	%  end
	%end
	%error('stop here')

end

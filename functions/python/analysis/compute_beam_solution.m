function [ solution ] = compute_beam_solution( problem, beam, constitutive )

%Solve static equilibrium equations
[ solution.u ] = FRANS_SolveStatic( beam.K, beam.p );

% Ku=f
% K'u+Ku'=f'
% Ku'=f'-K'u

%Solve the gradient of the solution
N = problem.info.num_sec;
NDof = size(beam.p,1);
NCase = size(beam.p,2);
edof = problem.frans.edof_1d;
for grd=1:problem.nvar
	solution.grad{grd}.u=zeros(NDof,NCase,N);
	for i=1:N
		grad_p=zeros(NDof,NCase);
		ue = solution.u(edof(:,i),:);
		grad_p(edof(:,i),:) = -beam.grad{grd}.Ke(:,:,i)*ue;
		if strcmp(problem.type,'tower_monopile')
			if grd==1 & i<=problem.info.mudline_elms
				grad_p=grad_p-beam.dKsoil(:,:,i)*solution.u;
			end
		end
		% apply boundary conditions
		grad_p(1:6,:)=zeros(6,NCase);
		[ solution.grad{grd}.u(:,:,i) ] = FRANS_SolveStatic( beam.K, grad_p );
	end
end

% global fd_data;
% if fd_data.state==1
% 	%fd_data.data= [ dstrain(1).grad{1} dstrain(2).grad{1} ; dstrain(1).grad{2} dstrain(2).grad{2} ];
% 	%fd_data.data
% 	fd_data.data=zeros(40,NDof);
% 	for row=1:20
% 		for col=1:NDof
% 			fd_data.data(row,col)=solution.grad{1}.u(col,1,row);
% 			fd_data.data(row+20,col)=solution.grad{2}.u(col,1,row);
% 		end
% 	end
% elseif fd_data.state==2
% 	fd_data.data=solution.u(:,1)';
% 	%fd_data.data
% elseif fd_data.state==3
% 	fd_data.data=solution.u(:,1)';
% 	%fd_data.data
% end


%Recover cross section forces and moments
n_case=size(solution.u,2);
for i=1:n_case
	[ solution.f(:,:,i) ]=FRANS_RecoverForcesAndMomentsAtReferencePoint_withElm(beam.Ke,beam.Me,solution.u(:,i),problem.frans);
end

% [ Ki Ks ] u = f0
% Ki*u = -Ks*u

% f = Ke*u
% f = Ke'*u+Ke*u'
% solve the gradient of the recovered forces
for grd=1:problem.nvar
	solution.grad{grd}.f=zeros(24,N,NCase,N);
	% loop over DVs
	for dv=1:N
		% loop over elements
		for elm=1:N
			due = solution.grad{grd}.u(edof(:,elm),:,dv);
			sln=zeros(24,n_case);
			% Ke*u'
			sln=sln+beam.Ke(:,:,elm)*due;
			% Ke'*u
			if elm==dv
				ue=solution.u(edof(:,elm),:);
				sln=sln+beam.grad{grd}.Ke(:,:,dv)*ue;
			end
			solution.grad{grd}.f(:,elm,:,dv)=sln(:,:);
		end
	end
end

%% global fd_data;
%% if fd_data.state==1
%% 	%fd_data.data= [ dstrain(1).grad{1} dstrain(2).grad{1} ; dstrain(1).grad{2} dstrain(2).grad{2} ];
%% 	%fd_data.data
%% 	fd_data.data=zeros(40,24);
%% 	for row=1:20
%% 		for col=1:24
%% 			                                        % load, elm, case, dv
%% 			fd_data.data(row   ,col)=solution.grad{1}.f(col,1,1,row);
%% 			fd_data.data(row+20,col)=solution.grad{2}.f(col,1,1,row);
%% 		end
%% 	end
%% elseif fd_data.state==2
%% 	fd_data.data=(solution.f(:,1,1))';
%% 	%fd_data.data
%% elseif fd_data.state==3
%% 	fd_data.data=(solution.f(:,1,1))';
%% 	%fd_data.data
%% end

%Solve eigenvalue problem
[ solution.eigfreq, solution.eigvec ] = FRANS_SolveModal( beam.eigK, beam.eigM );
solution.eigfreq=real(solution.eigfreq);
solution.eigvec=real(solution.eigvec);

end



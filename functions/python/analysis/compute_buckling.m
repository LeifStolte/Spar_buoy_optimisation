function [ eta, deta ] = compute_buckling( problem, design, solution)

%Extract cross section moments

N=problem.info.num_sec;
if strcmp(problem.type,'blade_structure')
	M = solution.f(4,:,1)'; %based on the flapwise load case
	H = problem.geometry.h;
	W = design.a;
	T = design.t + design.e;
	grad{1}.H=zeros(size(H,1),1);
	grad{1}.W=ones(size(W,1),1);
	grad{1}.T=zeros(size(T,1),1);
	grad{2}.H=zeros(size(H,1),1);
	grad{2}.W=zeros(size(W,1),1);
	grad{2}.T=ones(size(T,1),1);
	grad{3}.H=zeros(size(H,1),1);
	grad{3}.W=zeros(size(W,1),1);
	grad{3}.T=ones(size(T,1),1);
elseif strcmp(problem.type, 'tower_monopile')
	M = solution.f(4,:,1)'; %based on the first load case
	H = design.D;
	W = 0.25*pi*design.D;
	T = design.t;
	grad{1}.H=ones(size(H,1),1);
	grad{1}.W=0.25*pi*ones(size(W,1),1);
	grad{1}.T=zeros(size(T,1),1);
	grad{2}.H=zeros(size(H,1),1);
	grad{2}.W=zeros(size(W,1),1);
	grad{2}.T=ones(size(T,1),1);
else
	disp('problem type not implemented');
	error();
end

% Get the sign and the absolute
sgn=sign(M);
M=abs(M);

matdata = problem.materialdata(1);
E1 = matdata.E1;
E2 = matdata.E2;
nu12 = matdata.nu12;
G12 = matdata.G12;
nu21 = nu12 * E2/E1;

%Build material constitutive matrix Q
Q11 = E1 / (1-nu12*nu21);
Q22 = E2 / (1-nu12*nu21);
Q12 = nu12*E2 / (1-nu12*nu21);
Q66 = G12;
Q_star = Q12 + 2*Q66 + sqrt(Q11*Q22);

%Compute value
eta = 6*M./(pi^2*Q_star.*H).*W./(T.^3.0);
eta = eta';
deta_dH = -6*M./(pi^2*Q_star.*H.^2.0).*W./(T.^3.0);
deta_dW = 6*M./(pi^2*Q_star.*H.*T.^3.0);
deta_dT = -18.0*M./(pi^2*Q_star.*H).*W./(T.^4.0);
deta_dM = 6./(pi^2*Q_star.*H).*W./(T.^3.0);

%Compute gradient
for grd=1:problem.nvar

	%deta_dh_comp=diag( deta_dH.*grad{grd}.H );
	%deta_dw_comp=diag( deta_dW.*grad{grd}.W );
	%deta_dt_comp=diag( deta_dT.*grad{grd}.T );
	%tmp=zeros(N);
	%tmp(:,:)=solution.grad{grd}.f(4,:,1,:);
	%deta_dm_comp=(tmp')*diag(deta_dM)*diag(sgn);

	%deta.grad{grd}=deta_dh_comp+deta_dw_comp+deta_dt_comp+deta_dm_comp;

	deta.grad{grd} = diag( deta_dH.*grad{grd}.H + deta_dW.*grad{grd}.W + deta_dT.*grad{grd}.T );
	tmp=zeros(N);
	% dof-load, elm, load-case, dv
	tmp(:,:)=solution.grad{grd}.f(4,:,1,:);
	deta_dm_comp=(tmp')*diag(deta_dM)*diag(sgn);
	deta.grad{grd}=deta.grad{grd}+deta_dm_comp;
end

end


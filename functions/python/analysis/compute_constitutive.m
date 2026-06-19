function [ constitutive ] = compute_constitutive( problem, csprops )

%Cross section stiffness and mass matrix
N = problem.info.num_sec;
constitutive.Ks = zeros(6,6,N);
constitutive.Ms = zeros(6,6,N);
for s = 1:N
    [ constitutive.Ks(:,:,s) ] = compute_Ks( csprops.total.kGA(s), csprops.total.EA(s), csprops.total.EIx(s), csprops.total.EIy(s), csprops.total.GK(s) );
    [ constitutive.Ms(:,:,s) ] = compute_Ms( csprops.total.ms(s), csprops.total.Ix(s), csprops.total.Iy(s) );
end

%Gradients of cross section stiffness and mass matrix
[ constitutive.grad ] = compute_gradient_KsAndMs( problem, csprops );


end


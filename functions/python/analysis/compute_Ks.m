function [ Ks ] = compute_Ks( kGA, EA, EIx, EIy, GK )

%Compute cross section stiffness matrix
Ks = zeros(6);
Ks(1,1) = kGA;
Ks(2,2) = kGA;
Ks(3,3) = EA;
Ks(4,4) = EIx;
Ks(5,5) = EIy;
Ks(6,6) = GK;

end


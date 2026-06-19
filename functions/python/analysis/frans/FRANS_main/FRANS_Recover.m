function [ recres ] = FRANS_Recover( frans_utils, csprops, constitutive, beamres )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recover 1D results - forces and moments
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Recover forces and moments at reference point
[ recres.fm(:,:,1) ]=FRANS_RecoverForcesAndMomentsAtReferencePoint(...
    constitutive.Ks,constitutive.Ms,beamres.d,frans_utils);

%Determine forces and moments with respect to elastic and shear center
p1=[csprops.ShearX csprops.ShearY];
p2=[csprops.ElasticX csprops.ElasticY]; 
alpha=0;
[ recres.fm(:,:,2) ] = FRANS_TransformForceAndMoments( p1, p2, alpha, recres.fm(:,:,1) );

end


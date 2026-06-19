function [ frans_utils, res, mat, recres ] = FRANS_Calc1DAnalysis( frans_options, constitutive )

% Build arrays for FRANS
[ frans_utils ] = FRANS_Utils( frans_options );

%Perform static analysis
[ res.d, mat.K, mat.M, mat.p ]=FRANS_StaticAnalysis(frans_utils,constitutive.Ks,constitutive.Ms);

% Perform eigenfrequency analysis
[ res.freq, res.vec ]=FRANS_ModalAnalysis(frans_utils,constitutive.Ks,constitutive.Ms);

end


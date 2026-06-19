function FRANS_PlotBeamResults( savefig_flag, beamres, utils )

%% Plot displacements
fh = FRANS_PlotBeamDisplacements( savefig_flag, beamres.d,utils );

%% Plot forces and moment along beam assembly
fh = FRANS_PlotForcesAndMoments( savefig_flag, beamres.fm, utils);

%% Plot eigenmodes
fh = FRANS_PlotEigenVecs( savefig_flag, beamres.eigfreq, beamres.eigvec, utils );

end


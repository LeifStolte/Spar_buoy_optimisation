function [d,eigfreq,eigvec,Kstat,Keig,Meig,pstat]=FRANS_StaticAndModalAnalysis(utils,Ks,Ms)

%Assemble global beam stiffness matrix
fprintf(1,'> Started assembling FE matrices \n');
[K,M]=FRANS_Assemble(utils,Ks,Ms);

%Build load vector
[p]=FRANS_BuildLoad(utils);

%Enforce boundary conditions
[Kstat,pstat]=FRANS_Enforce(K,p,utils);
fprintf(1,'> Finished assembling FE matrices \n');

%Solve static equilibrium equations
fprintf(1,'> Started solving static equilibrium equations  \n');
[d]=FRANS_SolveStatic(Kstat,pstat);
fprintf(1,'> Finished solving static equilibrium equations  \n');

%Enforce boundary conditions
[Keig,Meig]=FRANS_EigenEnforce(K,M,utils);
fprintf(1,'> Finished assembling FE matrices \n');

%Solve eigenvalue problem
fprintf(1,'> Started solving eigenvalue problem  \n');
[eigfreq,eigvec]=FRANS_SolveModal(Keig,Meig);
fprintf(1,'> Finished solving eigenvalue problem  \n');


end
function [d,K,M,p]=FRANS_StaticAnalysis(utils,Ks,Ms)

%Assemble global beam stiffness matrix
fprintf(1,'> Started assembling FE matrices \n');
[K,M]=FRANS_Assemble(utils,Ks,Ms);

%Build load vector
[p]=FRANS_BuildLoad(utils);

%Enforce boundary conditions
[K,p]=FRANS_Enforce(K,p,utils);
fprintf(1,'> Finished assembling FE matrices \n');

%Solve static equilibrium equations
fprintf(1,'> Started solving static equilibrium equations  \n');
[d]=FRANS_SolveStatic(K,p);
fprintf(1,'> Finished solving static equilibrium equations  \n');

end
function [theta_ref]=...
    FRANS_RecoverForcesAndMomentsAtReferencePoint(Ks,Ms,d,utils)

%% Initializing arrays
theta_ref=zeros(utils.mdim_1d,utils.ne_1d);
Ksmat=zeros(6,6,utils.ne_1d);
Msmat=zeros(6,6,utils.ne_1d);

%% Check for size of Ks and Ms matrices
if(size(Ks,3) == utils.ne_1d || size(Ms,3) == utils.ne_1d )
    Ksmat=Ks;
    Msmat=Ms;
elseif(size(Ks,3) == 1 || size(Ms,3) == 1 )
    for i=1:utils.ne_1d
        Ksmat(:,:,i)=Ks;
        Msmat(:,:,i)=Ms;
    end
else
    fprintf(1,'Error in FRANS_Assemble: Number of Ks and Ms matrices does not match number of beam elements! \n');
    return
end

%% Determine distribution of bending moment and transverse forces
for e=1:utils.ne_1d
    %Evaluate element stiffness matrix
    [Ke,Me]=FRANS_B4L(e,Ksmat(:,:,e),Ksmat(:,:,e),utils);
    %Determine transverse forces and moments
    u_e=d(1+(e-1)*(utils.mdim_1d-6):utils.mdim_1d+(e-1)*(utils.mdim_1d-6),1);
    theta_ref(:,e)=Ke*u_e;
end

end
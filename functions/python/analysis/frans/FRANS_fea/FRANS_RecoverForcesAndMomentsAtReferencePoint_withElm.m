function [theta_ref]=...
    FRANS_RecoverForcesAndMomentsAtReferencePoint_withElm(Ke,Me,d,utils)

%% Initializing arrays
theta_ref=zeros(utils.mdim_1d,utils.ne_1d);

%% Determine distribution of bending moment and transverse forces
for e=1:utils.ne_1d
    %Determine transverse forces and moments
    u_e=d(1+(e-1)*(utils.mdim_1d-6):utils.mdim_1d+(e-1)*(utils.mdim_1d-6),1);
    theta_ref(:,e)=Ke(:,:,e)*u_e;
end

end

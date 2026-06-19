function [ J ] = FRANS_B4L_Jacobian( xx, e, utils )
%B4L_JACOBIAN Summary of this function goes here
%   Detailed explanation goes here

%Jacobian
Jvec(1)=(-0.5625*xx^2*3 +0.5625*xx^1*2 +0.0625);
Jvec(2)=(1.6875*xx^2*3  -0.5625*xx^1*2 -1.6875);
Jvec(3)=(-1.6875*xx^2*3 -0.5625*xx^1*2 +1.6875);
Jvec(4)=(0.5625*xx^2*3  +0.5625*xx^1*2 -0.0625);
xvec=[utils.pr_1d(1,e) utils.pr_1d(4,e) utils.pr_1d(7,e) utils.pr_1d(10,e)];
J=Jvec(1)*xvec(1)+Jvec(2)*xvec(2)+Jvec(3)*xvec(3)+Jvec(4)*xvec(4);

end


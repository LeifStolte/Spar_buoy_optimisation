function [ Ms ] = compute_Ms( ms, Ix, Iy )
%Compute the cross section mass matrix
xm = 0;
ym = 0;
Ixy = 0;

% Cross section ms matrix
Ms11=[ ms     0           0   ;
    0        ms        0   ;
    0        0           ms];
Ms12=[ 0          0      -ms*ym;
    0          0          ms*xm;
    ms*ym    -ms*xm   0      ];
Ms22=[ Ix     -Ixy        0      ;
    -Ixy     Iy        0      ;
    0       0          Ix+Iy];
Ms=[Ms11  Ms12;
    -Ms12 Ms22];

end



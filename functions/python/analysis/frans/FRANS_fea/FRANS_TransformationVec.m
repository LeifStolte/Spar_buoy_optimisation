function [ vec ] = FRANS_TransformationVec( vec,p1,p2,alpha )
%********************************************************
% File: FRANS_TransformationVec
%   Function to translate and rotate the vector of cross setion forces and
%   moments.
%
% Syntax:
%   [ vec ] = FRANS_TransformationVec( vec,p1,p2,alpha )
%
% Input:
%   utils :  Structure with input data, useful arrays, and
%              constants
%
% Output:
%   vec   :  6x1 vector of cross section forces and moments
%   p1    :  2x1 vector with x,y coordinates of shear center
%   p2    :  2x1 vector with x,y coordinates of elastic center
%   alpha :  rotation angle
%
% Calls:
%
% Revisions:
%   Version 1.0    07.02.2012   José Pedro Blasques
%
% (c) DTU Wind Energy
%********************************************************

%Matrix translation
T2=eye(6);
%T2(1,6)=p1(2);T2(2,6)=-p1(1);T2(3,4)=-p2(2);T2(3,5)=p2(1);
T2(6,1)=-p1(2);T2(6,2)=p1(1);
T2(4,3)=p2(2);T2(5,3)=-p2(1);

%Transform matrix'
vec=(T2\eye(6))*vec;

%Matrix rotation
c=cosd(alpha);
s=sind(alpha);

R11=[c   s  0;
    -s  c  0;
    0   0  1];
R1=[R11      zeros(3);
    zeros(3) R11];

vec=R1'*vec;

end
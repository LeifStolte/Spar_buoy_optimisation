function [ fm_transf ] = FRANS_TransformForceAndMoments( p1, p2, alpha, fm )

fm_transf=zeros(size(fm,1),1);
for e = 1:size(fm,2)
    [fm_transf(1:6,e)]=FRANS_TransformationVec(fm(1:6,e),p1,p2,alpha);
    [fm_transf(end-5:end,e)]=FRANS_TransformationVec(fm(end-5:end,e),p1,p2,alpha);
end

end


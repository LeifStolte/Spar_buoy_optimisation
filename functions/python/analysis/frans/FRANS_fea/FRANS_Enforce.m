function [K,p]=FRANS_Enforce(K,p,utils)

%Enforce boundary conditions on stiffness matrix
for i = 1:utils.nb_1d
    idof = utils.mdim_1d/utils.nnpe_1d*(utils.bc_1d(i,1)-1) + utils.bc_1d(i,2);
    for ii=1:utils.nlc_1d
		if ii<=size(p,2)
			p(1:end,ii)= p(1:end,ii) - K(1:end, idof)*utils.bc_1d(i,3);
			p(idof,ii) = utils.bc_1d(i, 3);
		end
    end
    K(:, idof) = 0.D0;
    K(idof, :) = 0.D0;
    K(idof, idof) = 1.D0;
end

end

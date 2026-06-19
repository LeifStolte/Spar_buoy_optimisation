function [p]=FRANS_BuildLoad(utils)

%Initialize load vector
p1d(1:utils.nn_1d*6,utils.nlc_1d)=0;
%Build load vector
for i = 1: utils.np_1d
    %Point load
    if (utils.f_1d(i, 2) == 1)
        %Determine the position of the global degree of freedom
        pposition = utils.mdim_1d/utils.nnpe_1d*(utils.f_1d(i,3)-1)+utils.f_1d(i,4);
        %Store the load value in the corresponding position
        p1d(pposition,utils.f_1d(i, 1))=p1d(pposition,utils.f_1d(i, 1))+utils.f_1d(i,5);
    end
    %Pressure loads
    if (utils.f_1d(i, 2) == 2)
        %Build element load vector
        paux=zeros(6,1);
        %Determine the position of the global degree of freedom
        pposition = utils.f_1d(i,4);
        %Store the load value in the corresponding position
        paux(pposition) = utils.f_1d(i,5);
        %Integrate loads
        [Npres]=FRANS_IntegrateLoad(i,utils);
        %pe=Npres'*Npres*paux;
        pe=Npres*paux;
        %Store everything in the load vector
        for e=1:utils.ne_1d
            for j = 1:utils.mdim_1d
                p1d(utils.edof_1d(j,e),utils.f_1d(i,1))=p1d(utils.edof_1d(j,e),utils.f_1d(i,1))+pe(j);
            end
        end
    end
    %Acceleration
    if (utils.f_1d(i, 2) == 3)
        %Integrate loads
        [Npres]=FRANS_IntegrateLoad(i,utils);
        paux=zeros(utils.mdim_1d,1);
        for ii=1:utils.nnpe_1d
            %Determine the position of the global degree of freedom
            pposition = utils.mdim_1d/utils.nnpe_1d*(ii-1)+utils.f_1d(i,4);
            %Store the load value in the corresponding position
            paux(pposition) = utils.f_1d(i,5);
        end
        %Build element load vector
        pe=Npres'*Npres*paux;
        %Store everything in the load vector
        for e=1:utils.ne_1d
            for j = 1:utils.mdim_1d
                p1d(utils.edof_1d(j,e),utils.f_1d(i,1))=p1d(utils.edof_1d(j,e),utils.f_1d(i,1))+pe(j);
            end
        end
    end
end
p=p1d;

    

end


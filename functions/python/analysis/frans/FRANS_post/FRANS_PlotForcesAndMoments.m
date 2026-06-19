function [fighandle] = FRANS_PlotForcesAndMoments(savefig_flag, fm,utils)


%% Gather F and M values for each of the element end nodes
plotFandM=zeros(6,2*utils.ne_1d,size(fm,3));
for v=1:size(fm,3)
    iv=0;
    for i=1:utils.ne_1d
        for ii=1:2
            iv=iv+1;
            for iii=1:6
                ipos=iii+(1-ii)*(1-utils.nnpe_1d)*6;
                plotFandM(iii,iv,v) = ((-1)^ii) * fm(ipos,i,v);
            end
        end
    end
end

%% Gather length coordinates for each of the element end nodes
zpos=zeros(utils.ne_1d+2,1);
i=0;
for e=1:utils.ne_1d
    i=i+1;
    zpos(i)=utils.nl_1d(utils.el_1d(e,2),2);
    i=i+1;
    zpos(i)=utils.nl_1d(utils.el_1d(e,utils.nnpe_1d+1),2);
end

%% Plot forces and moments
titlecell={'T_x','T_y','T_z','M_x','M_y','M_z'};
colormarker={'o-r', '-*g', '+-b'};
fighandle = figure;
for ii=1:size(plotFandM,3)
    for i=1:6
        subplot(2,3,i)
        hold on
        plot(zpos,plotFandM(i,:,ii),colormarker{ii})
        title(titlecell{i})
        xlabel('z')
        hold off
    end
end
end

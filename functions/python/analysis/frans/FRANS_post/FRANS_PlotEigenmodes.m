
function BECAS_1DPlotEigenmodes(eigfreq,eigvec)

linecolor={'-r','-g','-b','--r','--g','--b'};
xvec=1:size(eigvec,1)/6;
% [eigfreq,eigfreq_sort]=sort(sqrt(eigfreq),'ascend');
% eigfreq_sort
% eigvec=eigvec(:,eigfreq_sort);
figure(9)
for f=1:9
    subplot(3,3,f)
    hold on
    title(eigfreq(f))
    plot(xvec,eigvec(1:6:end,f),'-r')
    plot(xvec,eigvec(2:6:end,f),'-g')
    plot(xvec,eigvec(3:6:end,f),'-b')
    plot(xvec,eigvec(4:6:end,f),'-.r')
    plot(xvec,eigvec(5:6:end,f),'-.g')
    plot(xvec,eigvec(6:6:end,f),'-.b')
    % [AX,A1,A2] = plotyy(xvec,eigvec(1:6:end,f),xvec,eigvec(4:6:end,f),'plot');
    % [BX,B1,B2] = plotyy(xvec,eigvec(2:6:end,f),xvec,eigvec(5:6:end,f),'plot');
    % [CX,C1,C2] = plotyy(xvec,eigvec(3:6:end,f),xvec,eigvec(6:6:end,f),'plot');
    % set(A1,'LineStyle','-','Color','r')
    % set(A2,'LineStyle','-.','Color','r')
    % set(B1,'LineStyle','-','Color','g')
    % set(B2,'LineStyle','-.','Color','g')
    % set(C1,'LineStyle','-','Color','b')
    % set(C2,'LineStyle','-.','Color','b')
    hold off
end
figure(10)
for f=1:9
    subplot(3,3,f)
    hold on
    title(eigfreq(f))
    plot(xvec,eigvec(1:6:end,f),'-r')
    plot(xvec,eigvec(2:6:end,f),'-g')
    plot(xvec,eigvec(3:6:end,f),'-b')
    %   plot(xvec,eigvec(4:6:end,f),'-.r')
    %   plot(xvec,eigvec(5:6:end,f),'-.g')
    %   plot(xvec,eigvec(6:6:end,f),'-.b')
    % [AX,A1,A2] = plotyy(xvec,eigvec(1:6:end,f),xvec,eigvec(4:6:end,f),'plot');
    % [BX,B1,B2] = plotyy(xvec,eigvec(2:6:end,f),xvec,eigvec(5:6:end,f),'plot');
    % [CX,C1,C2] = plotyy(xvec,eigvec(3:6:end,f),xvec,eigvec(6:6:end,f),'plot');
    % set(A1,'LineStyle','-','Color','r')
    % set(A2,'LineStyle','-.','Color','r')
    % set(B1,'LineStyle','-','Color','g')
    % set(B2,'LineStyle','-.','Color','g')
    % set(C1,'LineStyle','-','Color','b')
    % set(C2,'LineStyle','-.','Color','b')
    hold off
end
figure(11)
for f=1:9
    subplot(3,3,f)
    hold on
    title(eigfreq(f))
    %   plot(xvec,eigvec(1:6:end,f),'-r')
    %   plot(xvec,eigvec(2:6:end,f),'-g')
    %   plot(xvec,eigvec(3:6:end,f),'-b')
    plot(xvec,eigvec(4:6:end,f),'-.r')
    plot(xvec,eigvec(5:6:end,f),'-.g')
    plot(xvec,eigvec(6:6:end,f),'-.b')
    % [AX,A1,A2] = plotyy(xvec,eigvec(1:6:end,f),xvec,eigvec(4:6:end,f),'plot');
    % [BX,B1,B2] = plotyy(xvec,eigvec(2:6:end,f),xvec,eigvec(5:6:end,f),'plot');
    % [CX,C1,C2] = plotyy(xvec,eigvec(3:6:end,f),xvec,eigvec(6:6:end,f),'plot');
    % set(A1,'LineStyle','-','Color','r')
    % set(A2,'LineStyle','-.','Color','r')
    % set(B1,'LineStyle','-','Color','g')
    % set(B2,'LineStyle','-.','Color','g')
    % set(C1,'LineStyle','-','Color','b')
    % set(C2,'LineStyle','-.','Color','b')
    hold off
end

end
% filename='Square_5_12_EigenModes';savefig(filename,'eps','-fonts','-crop','-c1','-r800')
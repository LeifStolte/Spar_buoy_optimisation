function fh = FRANS_PlotEigenVecs(savefig_flag,eigfreq,eigvec,utils)


linecolor={'-r','-g','-b','--r','--g','--b'};
xvec=utils.nl_1d(:,2);
fh=figure;
title('Beam eigenmodes')
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
    hold off
    legend('u_x','u_y','u_z','r_x','r_y','r_z','Orientation','Horizontal');
end
% filename='Square_5_12_EigenModes';savefig(filename,'eps','-fonts','-crop','-c1','-r800')
end


function fh = FRANS_PlotBeamDisplacements( savefig_flag,d,utils )

linecolor={'-r','-g','-b','--r','--g','--b'};
xvec=utils.nl_1d(:,2);
for lc=1:size(d,2)
    fh=figure;
    hold on
    plot(xvec,d(1:6:end,lc),'-r')
    plot(xvec,d(2:6:end,lc),'-g')
    plot(xvec,d(3:6:end,lc),'-b')
    plot(xvec,d(4:6:end,lc),'-.r')
    plot(xvec,d(5:6:end,lc),'-.g')
    plot(xvec,d(6:6:end,lc),'-.b')
    hold off
    title(['Displacements and rotations - Load case ' num2str(lc)]);
    legend('u_x','u_y','u_z','r_x','r_y','r_z','location','BestOutside');
end
end


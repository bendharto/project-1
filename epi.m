
ax = tic;
%    load('epi.mat');
  

[dat, p] = epi_pha_correctx(dat, p);

fprintf('epi_pha_correctx : %f seconds\n',toc(ax));

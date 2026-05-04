     %-------------------------------------------------------------------%
     %  script for illustrating cp_d1_lsx on labeling of 3D point cloud  %
     %-------------------------------------------------------------------%
% References:
% H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for Monotone 
% Inclusion and Convex Optimization, Optimization Letters, 2018, 1-24
%
% L. Landrieu and H. Raguet and B. Vallet and C. Mallet and M. Weinmann, A
% structured regularization framework for spatially smoothing semantic
% labelings of 3D point clouds, Journal of Photogrammetry and Remote Sensing,
% 2017, 132, 102-118 
%
% Hugo Raguet 2017, 2023
cd(fileparts(which('example_labeling_3D.m')));
addpath('./bin/', './doc/');

%%%  classes involved in the task  %%%
classNames = {'road', 'vegetation', 'facade', 'hardscape', ...
    'scanning artifacts', 'cars'};
classId = uint8(1:6)';

%%%  parameters; see octave/doc/pfdr_d1_lsx.m  %%%
options = struct; % reinitialize
options.dif_tol = 1e-4;
options.dif_it = 16;
options.it_max = 1e3;
options.rho = 1.5;
options.cond_min = 1e-1;
options.dif_rcd = 0.0;
options.verbose = options.dif_it;

%%%  initialize data  %%%
load('../data/labeling_3D.mat')
% convert graph representation from forward-star to adjacency list
V = length(first_edge) - 1;
edges = repelem(uint32(0:V-1), first_edge(2:end) - first_edge(1:end-1));
edges = [edges; adj_vertices'];
clear first_edge adj_vertices
options.edge_weights = homo_d1_weight;

% compute prediction performance of random forest
[~, ML] = max(y, [], 1);
F1 = zeros(1, length(classId));
for k=1:length(classId)
    predk = ML == classId(k);
    truek = ground_truth == classId(k);
    F1(k) = 2*sum(predk & truek)/(sum(predk) + sum(truek));
end
fprintf('\naverage F1 of random forest prediction: %.2f\n\n', mean(F1));
clear predk truek

%%%  solve the optimization problem  %%%
tic;
[x, Obj] = pfdr_d1_lsx(loss, y, edges, options);
time = toc;
fprintf('Total MEX execution time %.0f s\n\n', time);

% compute prediction performance of spatially regularized prediction
[~, ML] = max(x, [], 1);
F1 = zeros(1, length(classId));
for k=1:length(classId)
    predk = ML == classId(k);
    truek = ground_truth == classId(k);
    F1(k) = 2*sum(predk & truek)/(sum(predk) + sum(truek));
end
fprintf('\naverage F1 of spatially regularized prediction: %.2f\n\n', ...
    mean(F1));
clear predk truek

figure(1), clf, plot(linspace(0, time, length(Obj)), Obj);
title('objective evolution');
xlabel('time (s)');
ylabel(sprintf('KL^{(%.1f)}(y||x) + ||x||_{\\delta_{1,1}}', loss));

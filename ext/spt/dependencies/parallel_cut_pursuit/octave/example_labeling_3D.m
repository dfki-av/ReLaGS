     %-------------------------------------------------------------------%
     %  script for illustrating cp_d1_lsx on labeling of 3D point cloud  %
     %-------------------------------------------------------------------%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation, International Conference on
% Machine Learning, PMLR, 2018, 80, 4244-4253
%
% L. Landrieu et al., A structured regularization framework for spatially
% smoothing semantic labelings of 3D point clouds, ISPRS Journal of
% Photogrammetry and Remote Sensing, 132:102-118, 2017
%
% Hugo Raguet 2017, 2018, 2019, 2023
cd(fileparts(which('example_labeling_3D.m')));
addpath('./bin/', './doc/');

%%%  classes involved in the task  %%%
classNames = {'road', 'vegetation', 'facade', 'hardscape', ...
    'scanning artifacts', 'cars'};
classId = uint8(1:6)';

%%%  parameters; see octave/doc/cp_d1_lsx.m  %%%
options = struct; % reinitialize
options.cp_dif_tol = 1e-3;
options.K = 3;
options.balance_parallel_split = true;

%%%  initialize data  %%%
% For details on the data and parameters, see H. Raguet, A Note on the
% Forward-Douglas--Rachford Splitting for Monotone Inclusion and Convex
% Optimization Optimization Letters, 2018, 1-24
load('../pcd-prox-split/data/labeling_3D.mat')
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
options.compute_Obj = true;
options.compute_Time = true;
tic;
[Comp, rX, Obj, Time] = cp_d1_lsx(loss, y, first_edge, adj_vertices, options);
time = toc;
x = rX(:, Comp + 1); % rX is components values, Comp is components assignments
clear Comp rX;
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

figure(1), clf, plot(Time, Obj);
title('objective evolution');
xlabel('time (s)');
ylabel(sprintf('KL^{(%.1f)}(y||x) + ||x||_{\\delta_{0}}', loss));

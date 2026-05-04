            %-----------------------------------------------------%
            %  script for illustrating cp_d1_ql1b on EEG problem  %
            %-----------------------------------------------------%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation, International Conference on
% Machine Learning, PMLR, 2018, 80, 4244-4253
%
% Hugo Raguet 2017, 2018, 2019, 2023
cd(fileparts(which('example_EEG.m')));
addpath('./bin/', './doc/');

%%%  general parameters  %%%
plot_results = true;
print_results = false;

% parameters for colormap
numberOfColors = 256;
darkLevel = 1/16;

%%%  parameters; see octave/doc/cp_d1_ql1b.m %%%
options = struct; % reinitialize
options.cp_dif_tol = 1e-4;
options.cp_it_max = 15;
options.pfdr_rho = 1.5;
% options.pfdr_cond_min = 1e-2;
% options.pfdr_dif_rcd = 0.0;
options.pfdr_dif_tol = 1e-1*options.cp_dif_tol;
% options.pfdr_it_max = 1e4;
% options.verbose = 1e3;
% options.max_num_threads = 8;
options.balance_parallel_split = false;

%%%  initialize data  %%%
% dataset courtesy of Ahmad Karfoul and Isabelle Merlet, LTSI, INSERM U1099
% Penalization parameters computed with SURE methods, heuristics adapted from
% H. Raguet: A Signal Processing Approach to Voltage-Sensitive Dye Optical
% Imaging, Ph.D. Thesis, Paris-Dauphine University, 2014
load('../pcd-prox-split/data/EEG.mat')
options.edge_weights = d1_weights;
options.l1_weights = l1_weights;
options.low_bnd = 0.0;

%%%  solve the optimization problem  %%%
options.compute_Obj = true;
options.compute_Time = true;
tic;
[Comp, rX, Obj, Time] = cp_d1_ql1b(y, Phi, first_edge, adj_vertices, options);
time = toc;
x = rX(Comp + 1); % rX is components values, Comp is components assignment
clear Comp rX;
fprintf('Total MEX execution time %.1f s\n\n', time);

%%%  compute Dice scores and print results  %%%
% support retrieved with raw model
supp0 = x0 ~= 0; % ground truth support 
supp = x ~= 0;
DS = 2*sum(supp0 & supp)/(sum(supp0) + sum(supp));
% support retrieved by discarding nonsignificant values with 2-means clustering
abss = abs(x);
sabs = sort(abss);
n0 = 0; n1 = length(x0); % number of elements per cluster
sum0 = 0; sum1 = sum(sabs); % sum of each cluster
m = sum1/n1;
while 2*sabs(n0+1) < m
    n0 = n0 + 1;
    n1 = n1 - 1;
    sum0 = sum0 + sabs(n0);
    sum1 = sum1 - sabs(n0);
    m = (sum0/n0 + sum1/n1);
end
suppa = abss > (m/2);
DSa = 2*sum(supp0 & suppa)/(sum(supp0) + sum(suppa));
fprintf(['Dice score: raw %.2f; approx (discard less significant with ' ...
    '2-means) %.2f\n\n'], DS, DSa);

if plot_results
    % the following creates a colormap adapted to representation of sparse data
    n = numberOfColors/3;
    % luminance of pure red is 0.2989
    redMap = [linspace(darkLevel/0.2989, 1, floor(n))'; ...
              ones(round(n) + ceil(n), 1)];
    greenMap = [zeros(floor(n), 1);  (1:round(n))'/round(n); ones(ceil(n), 1)];
    blueMap = [zeros(floor(n) + round(n), 1); (1:ceil(n))'/ceil(n)];
    colMap = [redMap, greenMap, blueMap];
    colMap = [darkLevel*[1, 1, 1]; colMap]; % this is luminance of pure red
    clear redMap greenMap blueMap

    CAM = 1.0e+03*[-0.6329 1.5675 0.1686]; % camera parameter well adapted to
                                           % the sources distribution
    % get absolute min and max values to plot with same colormap
    x0min = min(x0); 
    x0max = max(x0);

    %% ground truth activity
    figure(1), clf, colormap(colMap);
    % map the color index
    xcol = floor((x0 - x0min)/(x0max - x0min)*numberOfColors) + 2;
    xcol(~supp0) = 1;
    % require octave 4.2.2 or later, fixing a bug in trisurf
    trisurf(mesh.f, mesh.v(:,1), mesh.v(:,2), mesh.v(:,3), xcol, ...
        'CDataMapping', 'direct');
    set(gca, 'Color', 'none'); axis off;
    set(gca, 'CameraPosition', CAM);
    drawnow('expose');
    if print_results
        fprintf('print ground truth... ');
        print(gcf, '-depsc', 'EEG_ground_truth');
        fprintf('done.\n');
    end

    %% retrieved activity
    figure(2), clf, colormap(colMap);
    xcol = floor((x - x0min)/(x0max - x0min)*numberOfColors) + 2;
    xcol(~supp) = 1;
    % be sure to run octave 4.2.2 or later, fixing a bug in trisurf
    trisurf(mesh.f, mesh.v(:,1), mesh.v(:,2), mesh.v(:,3), xcol, ...
        'CDataMapping', 'direct');
    set(gca, 'Color', 'none'); axis off;
    set(gca, 'CameraPosition', CAM);
    drawnow('expose');
    if print_results
        fprintf('print retrieved brain activity... ');
        print(gcf, '-depsc', 'EEG_brain_activity');
        fprintf('done.\n');
    end

    %% retrieved support
    figure(3), clf, colormap(colMap);
    xcol = 1 + suppa*numberOfColors;
    % be sure to run octave 4.2.2 or later, fixing a bug in trisurf
    trisurf(mesh.f, mesh.v(:,1), mesh.v(:,2), mesh.v(:,3), xcol, ...
        'CDataMapping', 'direct');
    set(gca, 'Color', 'none'); axis off;
    set(gca, 'CameraPosition', CAM);
    drawnow('expose');
    if print_results
        fprintf('print retrieved brain sources... ')
        print(gcf, '-depsc', 'EEG_brain_sources');
        fprintf('done.\n');
    end

    figure(4), clf, semilogy(Time, Obj);
    title('objective evolution');
    xlabel('time (s)');
    ylabel('1/2 ||y - \Phi x||^2 + ||x||_{\ell_1} + ||x||_{\delta_1}');
end

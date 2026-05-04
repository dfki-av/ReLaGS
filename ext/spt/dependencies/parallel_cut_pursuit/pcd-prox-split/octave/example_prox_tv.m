        %-------------------------------------------------------------%
        %  script for illustrating pfdr_prox_tv on denoising problem  %
        %-------------------------------------------------------------%
% Reference:
% H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
% Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
% Sciences, 2015, 8, 2706-2739
%
% Hugo Raguet 2022, 2023
cd(fileparts(which('example_prox_tv.m')));
addpath('./bin/', './doc/');
addpath('../grid-graph/octave/bin/', '../grid-graph/octave/doc/');
grid_graph_bin_content = what('../grid-graph/octave/bin/');
if (isempty(grid_graph_bin_content) || isempty(grid_graph_bin_content(1).mex))
    error(['CP prox TV example: compile grid-graph utilities in ' ...
        'grid-graph/octave/']);
end

%%%  general parameters  %%%
plot_results = true;
print_results = false;
image_file = '../data/Lola.jpg'
noise_level = .2
la_tv = .5*noise_level;

%%%  parameters; see octave/doc/pfdr_prox_tv.m %%%
options = struct; % reinitialize
options.d1p = 2;
options.rho = 1.5;
options.cond_min = 1e-1;
options.dif_rcd = 0.0;
options.dif_tol = 1e-3;
options.it_max = 1e3;
options.verbose = 1e2;

%%%  initialize data  %%%
fprintf('load image and apply transformation... ');
x0 = imread(image_file);
x0 = x0(1:3:end, 1:3:end, :); % some downsampling
x0 = single(x0)/255;
y = x0 + noise_level*randn(size(x0), 'single');
fprintf('done.\n');

%% plot the observations
if plot_results
    figure(1), clf, colormap('gray');
    subplot(1, 3, 1), imagesc(x0);
    set(gca, 'xtick', [], 'ytick', []); axis image; title('Original');
    subplot(1, 3, 2), imagesc(max(0, min(y, 1)));
    set(gca, 'xtick', [], 'ytick', []); axis image;
    title(sprintf('Noisy (%.2f dB)', ...
        10*log10(sum(x0(:).^2)/sum((y(:) - x0(:)).^2))));
    drawnow('expose');
end

fprintf('generate adjacency graph... ');
[edges, connectivities] = grid_to_graph(uint32(size(y, [1 2])), 2, false);
options.edge_weights = zeros([1 length(connectivities)], class(y));
options.edge_weights(connectivities == 1) = la_tv;
options.edge_weights(connectivities == 2) = la_tv/sqrt(2);
clear connectivities
fprintf('done.\n');

%%%  solve the optimization problem  %%%
tic;
[x, Obj] = pfdr_prox_tv(reshape(y, prod(size(y, [1 2])), size(y, 3))', ...
    edges, options);
time = toc;
x = reshape(x', size(x0));
fprintf('Total MEX execution time %.1f s\n\n', time);

%%  plot results
if plot_results
    figure(1)
    subplot(1, 3, 3), imagesc(max(0, min(x, 1)));
    set(gca, 'xtick', [], 'ytick', []); axis image;
    title(sprintf('PFDR prox TV (%.2f dB)',
        10*log10(sum(x0(:).^2)/sum((x(:) - x0(:)).^2))));
    drawnow('expose');
    if print_results
        fprintf('print results... ');
        print(gcf, '-depsc', sprintf('%s/images.eps', experiment_dir));
        fprintf('done.\n');
    end

    figure(2), clf, plot(linspace(0, time, length(Obj)), Obj);
    title('objective evolution');
    xlabel('time (s)');
    ylabel('1/2 ||y - x||^2 + ||x||_{\delta_{1,2}}');
end

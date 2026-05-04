origDir = pwd; % remember working directory
cd(fileparts(which('compile_grid_graph_mex.m'))); 
if ~exist('./bin/'), mkdir('bin/'); end
try
    % compilation flags 
    [~, CXXFLAGS] = system('mkoctfile -p CXXFLAGS');
    [~, LDFLAGS] = system('mkoctfile -p LDFLAGS');
    % some versions introduces a newline character (10)
    % in the output of 'system'; this must be removed
    if CXXFLAGS(end)==10, CXXFLAGS = CXXFLAGS(1:end-1); end
    if LDFLAGS(end)==10, LDFLAGS = LDFLAGS(1:end-1); end
    CXXFLAGSorig = CXXFLAGS;
    LDFLAGSorig = LDFLAGS;
    CXXFLAGS = [CXXFLAGS ' -Wextra -Wpedantic -std=c++11 -fopenmp -g0 ' ...
        '-DMIN_OPS_PER_THREAD=10000'];
    LDFLAGS = [LDFLAGS ' -fopenmp'];
    setenv('CXXFLAGS', CXXFLAGS);
    setenv('LDFLAGS', LDFLAGS);

    mex -I../include mex/grid_to_graph_mex.cpp ../src/grid_to_graph.cpp ...
        ../src/edge_list_to_forward_star.cpp ...
        -output bin/grid_to_graph
    clear grid_to_graph

    mex -I../include mex/edge_list_to_forward_star_mex.cpp ...
        ../src/edge_list_to_forward_star.cpp ...
        -output bin/edge_list_to_forward_star
    clear edge_list_to_forward_star

    if exist('edge_list_to_forward_star_mex.o'), system('rm *.o'); end
catch % if an error occur, makes sure not to change the working directory
    % back to original environment
    setenv('CXXFLAGS', CXXFLAGSorig);
    setenv('LDFLAGS', LDFLAGSorig);
    cd(origDir);
	rethrow(lasterror);
end
% back to original environment
setenv('CXXFLAGS', CXXFLAGSorig);
setenv('LDFLAGS', LDFLAGSorig);
cd(origDir);

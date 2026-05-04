origDir = pwd; % remember working directory
cd(fileparts(which('compile_pcd_prox_split_mex.m'))); 
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
    % MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
    CXXFLAGS = [CXXFLAGS ' -Wextra -Wpedantic -std=c++11 -fopenmp -g0 ' ...
        '-DMIN_OPS_PER_THREAD=10000'];
    LDFLAGS = [LDFLAGS ',-fopenmp'];
    setenv('CXXFLAGS', CXXFLAGS);
    setenv('LDFLAGS', LDFLAGS);

    %{
    mex -I../include -I../matrix-tools/include mex/pfdr_d1_ql1b_mex.cpp ...
        ../src/pfdr_d1_ql1b.cpp ../src/pfdr_graph_d1.cpp ...
        ../src/pcd_fwd_doug_rach.cpp ../src/pcd_prox_split.cpp ...
        ../matrix-tools/src/matrix_tools.cpp ...
        -output bin/pfdr_d1_ql1b
    clear pfdr_d1_ql1b
    %}

    %{
    mex -I../include -I../proj-simplex/include mex/pfdr_d1_lsx_mex.cpp ...
        ../src/pfdr_d1_lsx.cpp ../src/pfdr_graph_d1.cpp ...
        ../src/pcd_fwd_doug_rach.cpp ../src/pcd_prox_split.cpp ...
        ../proj-simplex/src/proj_simplex.cpp ...
        -output bin/pfdr_d1_lsx
    clear pfdr_d1_lsx
    %}

    % %{
    mex -I../include mex/pfdr_prox_tv_mex.cpp ...
        ../src/pfdr_prox_tv.cpp ../src/pfdr_graph_d1.cpp ...
         ../src/pcd_fwd_doug_rach.cpp ../src/pcd_prox_split.cpp ...
        -output bin/pfdr_prox_tv
    clear pfdr_prox_tv
    %}
    
    if exist('pcd_prox_split.o'), system('rm *.o'); end
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

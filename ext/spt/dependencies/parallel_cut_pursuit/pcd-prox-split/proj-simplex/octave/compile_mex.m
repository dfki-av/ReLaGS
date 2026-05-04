origDir = pwd; % remember working directory
cd(fileparts(which('compile_mex.m'))); 
mkdir('bin/');
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
    % MIN_OPS_PER_THREAD roughly controls parallelization
    CXXFLAGS = [CXXFLAGS ' -Wextra -Wpedantic -std=c++11 -fopenmp -g0 ' ...
        '-DMIN_OPS_PER_THREAD=10000'];
    LDFLAGS = [LDFLAGS ',-fopenmp'];
    setenv('CXXFLAGS', CXXFLAGS);
    setenv('LDFLAGS', LDFLAGS);

    mex -I../include mex/proj_simplex_mex.cpp ../src/proj_simplex.cpp ...
        -output bin/proj_simplex_mex
    clear proj_simplex_mex

    if exist('proj_simplex.o'), system('rm *.o'); end
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

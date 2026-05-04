origDir = pwd; % remember working directory
cd(fileparts(which('compile_mex.m'))); 
if ~exist('bin/'), mkdir('bin/'); end
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
    % _GLIBCXX_PARALLEL is only useful for libstdc++ users
    % MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
    CXXFLAGS = [CXXFLAGS ' -Wextra -Wpedantic -Wno-sign-compare -std=c++11' ...
        ' -fopenmp -g0 -D_GLIBCXX_PARALLEL -DMIN_OPS_PER_THREAD=10000'];
    LDFLAGS = [LDFLAGS ' -fopenmp'];
    setenv('CXXFLAGS', CXXFLAGS);
    setenv('LDFLAGS', LDFLAGS);

    mex mex/wth_element_mex.cpp -output bin/wth_element

    clear wth_element

    if exist('wth_element.o'), system('rm *.o'); end
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

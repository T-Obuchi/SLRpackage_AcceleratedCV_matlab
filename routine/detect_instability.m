function [flag_inst]=detect_instability(dataV,errV)
%--------------------------------------------------------------------------
% detect_instability.m: 
%--------------------------------------------------------------------------
%
% DESCRIPTION:
%    Detect instability region of the approximate cross-validation (CV) formula 
%    for the cases of SCAD and MCP. 
%
% USAGE:
%    [flag_instability,MASK,cMASK]=detect_instability(dataV,errV);
%
% INPUT ARGUMENTS:
%    dataV       A vector of CV errors. 
%
%    errV        A vector of error bars of the CV errors (same dimension as dataV).
%
% OUTPUT ARGUMENTS:
%    flag_inst   Instability flag vector of the approximate CV datapoints. 
%                (0: stable, 1: unstable). The same dimension as dataV. 
%                If a component of flag_inst is 1, the corresponding component of dataV 
%                is not reliable and should be discarded. 
%
% DETAILS:
%    Detection method is rather simple. Just compute the difference 
%    of some neighboring components of dataV and compare it with the error
%    bar size stored in errV. If the difference is large enough compared to the
%    error bar, then the correponding datapoint is judged to be "unstable". 
%    See the source code 'detect_outliers.m' for the actual implementation.
%
% REFERENCES:
%    [1] Tomoyuki Obuchi and Ayaka Sakata: Cross validation in sparse linear 
%        regression with piecewise continuous nonconvex penalties and its acceleration
%        arXiv:1902.10375
%
%    [2] Tomoyuki Obuchi and Yoshiyuki Kabashima: Cross validation in LASSO 
%        and its acceleration
%        J. Stat. Mech. (2016) 053304
%
% DEVELOPMENT:
%    2 Mar. 2019: Original version was written.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ldata=length(dataV);
[MASK,cMASK]=detect_outliers(dataV,errV);
flag_inst=cMASK;    
loc=min(find(cMASK));
flag_inst(loc:Ldata)=1;
end

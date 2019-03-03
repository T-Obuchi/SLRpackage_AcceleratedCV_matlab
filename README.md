# SLRpackage_AcceleratedCV_matlab
Sparse linear regression package with accelerated cross-validation (CV) under
regularizations of L1 penalty (LASSO) or piecewise continuous nonconvex penalties.
Two piecewise continuous nonconvex penalties,
smoothly clipped absolute deviation (SCAD) [1] and minimax concave penalty (MCP) [2],
are treated.

This is free software, you can redistribute it and/or modify it under the terms of
the GNU General Public License, version 3 or above. See LICENSE for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


# DESCRIPTION
Compute a solution path with respect to amplitude parameter *lambda*
and evaluate the associated CV error (cve) using an efficient approximate formula from [3,4].
In particular, we solve the following problem:
<!-- ```math
   \hat{\bm{\beta}}=argmin_{\beta}( (1/2)||Y-X*\beta||_2^2 + \sum_{i}^{N}J(\beta_i;\eta) )
``` -->
<img src="https://latex.codecogs.com/gif.latex?\hat{\bm{\beta}}={\rm&space;argmin}_{\beta}\left(&space;(1/2)||Y-X\bm{\beta}||_2^2&space;&plus;&space;\sum_{i}^{N}J(\beta_i;\eta)&space;\right)" />

where
<img src="https://latex.codecogs.com/gif.latex?J(\beta;\eta)" />
is a regularizer and
<img src="https://latex.codecogs.com/gif.latex?\eta" />
 is the set of regularization parameters.
For SCAD and MCP,
<img src="https://latex.codecogs.com/gif.latex?\eta=\{\lambda,a\}" />
and a solution path  w.r.t.
<img src="https://latex.codecogs.com/gif.latex?\lambda" />
given *a* is computed.
For LASSO,
<img src="https://latex.codecogs.com/gif.latex?\eta=\lambda" />
and
<img src="https://latex.codecogs.com/gif.latex?J(\beta;\lambda)=\lambda|\beta|" />
the solution path is again w.r.t.
<img src="https://latex.codecogs.com/gif.latex?\lambda" />
.
We call
<img src="https://latex.codecogs.com/gif.latex?\lambda" />
as amplitude parameter and *a* as switching parameter according to [3].

The solution path is here obtained by the
<img src="https://latex.codecogs.com/gif.latex?\lambda" /> annealing [3].
This means that in the cases of SCAD and MCP a very particular solution path
is obtained in the multiple solution region appearing due to the nonconvexity of the penalties.
We recommend not to use the multiple solution region
which corresponds to the unstable region of the approximate CV formula
and is detected by *fig.stab* in the output of the package.

As option for CV, approximate and literal CVs are available,
and the default setting is the approximate one.


# REQUIREMENT AND PREPARATION
As a subroutine, cyclic coordinate descent (CCD) algorithm implemented by C language is used,
and hence please prepare your own C compiler connected to matlab.
Mex source files are provided. To compile them, pleae move to the "routine" folder and type
```matlab
    mex CCD_LASSO.c
    mex CCD_SCAD.c
    mex CCD_MCP.c
```
on matlab of your own environment. Please complete this before using the functions in the package.


# USAGE FOR SCAD
For the case of SCAD, to obtain a solution path, please use the function 'scadpath' in the "routine" folder with appropriate arguments. The actual usage is as follows:
```matlab
    fit = scadpath(Y,X,a)
    fit = scadpath(Y,X,a,lambdaV)
    fit = scadpath(Y,X,a,lambdaV,cvoption,kfold,theta,iter_max,beta_in,seed)
    (Use [] to apply the default value, e.g.
     fit = scadpath(Y,X,a,lambdaV,cvoption,[],theta,iter_max),
     fit = scadpath(Y,X,a,lambdaV,[],[],[],[],beta_in),
    )
```
Inputs:
- *Y*:          Response vector (M dimensional vector).

- *X*:          Matrix of covariates (M*N dimensional matrix).

- *a*:          Switching parameter (a real number larger than unity).   

- *lambdaV*:    Set of amplitude parameter (Llam dimensional vector). If not specified, a default set of Llam=100 is given.   

- *cvoption*:   Option for CV. Three options are available:
                'approximate': Approximate CV is conducted by the method of [1];
                'literal': Literal CV is conducted;
                'none': No CV is conducted.
The default setting is 'approximate'.

- *kfold*:      Fold number for CV. Only used when cvoption='literal'. The default value is 10.

- *theta*:      Threshold to judge the convergence of the CCD algorithm. The default value is 10^(-10).

- *iter_max*:   MAX iteration steps of the CCD algorithm. The default value is 10^5.

- *beta_in*:    Initial estimate of mean value of covariates' coefficients (N dimensional vector). Given as a zero vector if not specified.

- *seed*:       Seed for random number generation (positive integer). The default value is 1.

Outputs:
- *fit*:        A structure.

- *fit.lambda*: Set of amplitude parameter actually used in the pathwise estimation (Llam dimensional vector).

- *fit.beta*:   Estimated solution path (regression coefficients along the set of lambda, N*Llam dimensional matrix).

- *fit.conv*:   Flags checking convergence of the CCD algorithm at each lambda (Llam dimensional vector). (0: converged, 1: not converged).

- *fit.tre*:    Training error for the solution path (Llam dimensional vector).

- *fit.cve*:    CV error and its error bar for the solution path (Llam*2 dimensional matrix). fit.cve(:,1) gives the cve values and fit.cve(:,2) yields the error bars. When cvoption='none', an empty array is returned.

- *fit.stab*:   Flag for checking stability of the approximate CV (0: unstable, 1: stable). If the flag is zero, the corresponding approximate cve is not reliable and should be discarded. When cvoption='approximate', an Llam dimensional logical vector associated to all datapoints is returned, while when cvoption='literal' or 'none', an empty array is returned.

- *fit.time*:   Elapsed time for estimating solution path (fit.time(1)) and for computing cve (fit.time(2)).

A solution path, the training error, and the CV error are obtained as
*fit.beta*, *fit.tre*, and *fit.cve*, respectively. If you apply the approximate CV,
there is an instability point w.r.t. *lambda* below which the values of cve are not reliable.
The stable region of *lambda* is indicated *fit.stab*.
We recommend not to use the solution in the unstable region, because the instability is connected to the multiplicity of solutions owing to the nonconvexity of the penalty.

For more details, type 'help scadpath'. For the theoretical background of the approximate CV formula, see [3].


# USAGE FOR MCP
Just replace 'scadpath' by 'mcppath' in the SCAD usage.


# USAGE FOR LASSO
The switching parameter *a* is absent for LASSO, and
the approximate CV formula is stable and has two different options [4].
The usage thus becomes as follows:
```matlab
    fit = lassopath(Y,X)
    fit = lassopath(Y,X,lambdaV)
    fit = lassopath(Y,X,lambdaV,cvoption,kfold,theta,iter_max,beta_in,seed)
    (Use [] to apply the default value, e.g.
     fit = lassopath(Y,X,lambdaV,cvoption,[],theta,iter_max),
     fit = lassopath(Y,X,lambdaV,[],[],[],[],beta_in),
    )
```
Inputs:
- *Y*:          Response vector (M dimensional vector).

- *X*:          Matrix of covariates (M*N dimensional matrix).

- *lambdaV*:    Set of amplitude parameter (Llam dimensional vector). If not specified, a default set of Llam=100 is given.   

- *cvoption*:   Option for CV. Four options are available:
                'approximate1': Approximate CV based on 'approximation 1' in [4] is conducted;
                'approximate2': Approximate CV based on 'approximation 2' in [4] is conducted;
                'literal': Literal CV is conducted;
                'none': No CV is conducted.
The default setting is 'approximate2'.

- *kfold*:      Fold number for CV. Only used when cvoption='literal'. The default value is 10.

- *theta*:      Threshold to judge the convergence of the CCD algorithm. The default value is 10^(-10).

- *iter_max*:   MAX iteration steps of the CCD algorithm. The default value is 10^5.

- *beta_in*:    Initial estimate of mean value of covariates' coefficients (N dimensional vector). Given as a zero vector if not specified.

- *seed*:       Seed for random number generation (positive integer). The default value is 1.

Outputs:
- *fit*:        A structure.

- *fit.lambda*: Set of amplitude parameter actually used in the pathwise estimation (Llam dimensional vector).

- *fit.beta*:   Estimated solution path (regression coefficients along the set of lambda, N*Llam dimensional matrix).

- *fit.conv*:   Flags checking convergence of the CCD algorithm at each lambda (Llam dimensional vector). (0: converged, 1: not converged).

- *fit.tre*:    Training error for the solution path (Llam dimensional vector).

- *fit.cve*:    CV error and its error bar for the solution path (Llam*2 dimensional matrix). fit.cve(:,1) gives the cve values and fit.cve(:,2) yields the error bars. When cvoption='none', an empty array is returned. T

- *fit.time*:   Elapsed time for estimating solution path (fit.time(1)) and for computing cve (fit.time(2)).

A solution path, the training error, and the CV error are again obtained as
*fit.beta*, *fit.tre*, and *fit.cve*, respectively.
In the case of LASSO, two approximations called 'approximation 1' and 'approximation 2'
are available. The 'approximation 1' is more widely applicable irrespectively of
the choice of covariates matrix *X* while the 'approximation 2' is more robust and faster.
Please choose one of them according to your purpose and situation.
For more details, type 'help lassopath'.


# USAGE FOR APPROXIMATE CROSS-VALIDATION ONLY
The approximate CV formulas of [3,4] can be used independently of the optimization step. Suppose you have a SCAD estimator *beta* at given *lambda* and *a* (you may use any algorithm to obtain it). You can thus compute the associated cve by a function 'acv_scad' in the "routine" folder as
```matlab
    [acve aerr]=acv_scad(beta,Y,X,a,lambda);
```
where *acve* is the estimated cve value and *aerr* is its error bar. If you have a solution path w.r.t. *lambda*, it is better to save the approximate cve values and the error bars as vectors, because it is convenient to use a function 'detect_instability' in the "routine" folder to detect the instability point of the approximation. An example code is as follows:
```matlab
acveV=zeros(Llam,1);
aerrV=zeros(Llam,1);
for i=1:Llam
    [acve aerr]=acv_scad(beta(:,i),Y,X,a,lambdaV(i));
    acveV(i)=acve;
    aerrV(i)=aerr;
end
    [flag_unstable]=detect_instability(acveV,aerrV);
    flag_stable=not(flag_unstable);
```
*flag_stable* corresponds to *fit.stab* in the output of 'scadpath'. Note that this instability detection function works only when datapoints w.r.t. *lambda* are sufficiently taken.

For MCP, replace 'acv_scad' by 'acv_mcp'. The usage is identical.

For LASSO, 'approximation 1' is implemented as 'acv_lasso' and 'approximation 2' is implemented as 'saacv_lasso'.
The usage is
```matlab
    [acve aerr]=acv_lasso(beta,Y,X);
    [acve aerr]=saacv_lasso(beta,Y,X);
```
On the contrary to the MCP and SCAD cases, the argument does not *lambda*.


# DEMONSTRATION
In the "demo" folder, demonstration codes for scadpath, mcppath, and, lassopath (demo_scadpath.m, demo_mcppath.m, and demo_lassopath.m, respectively) are available. A demo code for comparing solution paths of these three regularizers, demo_comppath.m, is also provided.


# REFERENCE
[1] Jianqing Fan and Runze Li: Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties,
    Journal of the American statistical Association, 96, 1348, (2001)

[2] Cun-Hui Zhang: Nearly unbiased variable selection under minimax concave penalty,
    The Annals of Statistics, 938, 894, (2010)

[3] Tomoyuki Obuchi and Ayaka Sakata: Cross validation in sparse linear
    regression with piecewise continuous nonconvex penalties and its acceleration,
    arXiv:1902.10375

[4] Tomoyuki Obuchi and Yoshiyuki Kabashima: Cross validation in LASSO and its acceleration, J. Stat. Mech. 053304, (2016)

/*
*  CCD_MCP.c - Cyclic Coordinate Discent in MATLAB External Interfaces
*
***** Input ***** 
* 1: y (M dim.vector) ... observation
* 2: A (M times N matrix)and outputs a 1xN matrix (outMatrix) ... predictor
* 3: x_ini (N dim.vector) ... initial condition of coefficients
* 4: lambda (scalar) ... MCP parameter
* 5: a (scalar) ... MCP parameter
* 6: theta (scalar) ... for convergence check
* 7: iter_max (scalar) ... maximum iteration
*
***** output *****
* 1: x (N dim. vector) ... estimated coefficients
*    x[N] is for the convergence check.
*    if x[N]=0, CCD converges, x[N]=1, CCD is not converged.
*
*****
* The calling syntax is:
*
*		outMatrix = CCD_MCP(multiplier, inMatrix)
*
* This is a MEX file for MATLAB.
*/

#include "mex.h"

mwSize N;
mwSize M;
mwSize iter_max;
double lambda;
double a;
double theta;

double absv(double a)
{
    if(a >= 0){
        return(a);
    }else{
        return(-a);
    }
}

double sgn(double a)
{
    if(a >= 0){
        return(1);
    }else{
        return(-1);
    }
}

void shuffle(mwSize ary[])
{
    mwSize i, j, t;
    
    for(i = 0; i < N; i++)
    {
        j = rand()%N;
        t = ary[i];
        ary[i] = ary[j];
        ary[j] = t;
    }
}

/* The computational routine */
void CCD_MCP(double *y, double *A, double *x_ini, double *x)
{
    mwSize i, j, mu, t, x_diff;
    mwSize *i_perm;
    double tmp;
    double *r, *y_tmp, *x_old;
    
    i_perm = (mwSize *)malloc(sizeof(mwSize)*N);
    r = (double *)malloc(sizeof(double)*M);
    y_tmp = (double *)malloc(sizeof(double)*M);
    x_old = (double *)malloc(sizeof(double)*N);
    
    /* initial residue */
    for(mu = 0; mu < M; mu++){
        y_tmp[mu] = 0;
        for(i = 0; i < N; i++){
            y_tmp[mu] += A[mu*N+i]*x_ini[i];
        }
        r[mu] = y[mu] - y_tmp[mu];
    }
    /* initialization */
    t = 0;
    for(i = 0; i < N; i++){
        x_old[i] = x_ini[i];
        x[i] = x_ini[i];
        i_perm[i] = i;
    }
    x_diff = 0;
    
    while(x_diff != N && t < iter_max){
        t++;
        x_diff = 0;
        shuffle(i_perm);
        for(j = 0; j < N; j++){
            i = i_perm[j];
            tmp = x[i];
            for(mu = 0; mu < M; mu++){
                tmp += A[mu*N+i]*r[mu];
            }

            /* MCP thresholding */
            x_old[i] = x[i];
            x[i] = sgn(tmp)*(absv(tmp)-lambda)/(1.0-1.0/a)
	      *(absv(tmp)>lambda)*(absv(tmp)<=a*lambda);
	    x[i] += tmp*(absv(tmp)>a*lambda);
	    /* printf("%f\n", absv(tmp)-lambda); */
            x_diff += (absv(x_old[i]-x[i]) < theta);
            /* update of residue */
            for(mu = 0; mu < M; mu++){
                y_tmp[mu] += A[mu*N+i]*(x[i]-x_old[i]);
                r[mu] = y[mu] - y_tmp[mu];
            }
        }
    }
    x[N] = (t == iter_max);
     /* fprintf(fp, "%d %d\n", t, x[N]);*/
    free(r);
    free(y_tmp);
    free(x_old);
/*    fclose(fp);*/
}

/* The gateway function */
void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[])
{
    mwSize i, mu;
    double *y, *A, *x_ini, *x;
    
    /* get data */
    y = mxGetPr(prhs[0]);
    A = mxGetPr(prhs[1]);
    x_ini = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    a = mxGetScalar(prhs[4]);
    theta = mxGetScalar(prhs[5]);
    iter_max = (mwSize)mxGetScalar(prhs[6]);

    /* get dimension of the input matrix */
    M = (mwSize)mxGetM(prhs[0]);
    N = (mwSize)mxGetM(prhs[2]);
  
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,N+1,mxREAL);
    /* get a pointer to the real data in the output matrix */
    x = mxGetPr(plhs[0]);
  
    /* check for proper number of arguments */
    if(nrhs != 7) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
  
    /* call the computational routine */
    CCD_MCP(y, A, x_ini, x);
}

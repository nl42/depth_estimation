/** file:        imsmooth.c
 ** author:      Andrea Vedaldi
 ** description: Smooth an image.
 **/

/* AUTORIGHTS
Copyright (C) 2006 Andrea Vedaldi
      
This file is part of VLUtil.

VLUtil is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#include"mex.h"
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<assert.h>

#define greater(a,b) ((a) > (b))
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define PAD_BY_CONTINUITY

const double win_factor = 1.5 ;
const int nbins = 36 ;

/* convolve and transopose */
void
convolve(double*       dst_pt, 
         const double* src_pt,    int M, int N,
         const double* filter_pt, int W)
{
  int i,j ;
  for(j = 0 ; j < N ; ++j) {
    for(i = 0 ; i < M ; ++i) {
      double const* start = src_pt + max(i-W,0  ) ;        
      double const* stop  = src_pt + min(i+W,M-1) + 1 ;
      double const* g     = filter_pt + max(0, W-i) ;
      double acc = 0.0 ;
      while(stop != start) acc += (*g++) * (*start++) ;
      *dst_pt = acc ;
      dst_pt += N ;
    }
    src_pt += M ;
    dst_pt -= M*N - 1 ;
  }
}

/* convolve and transopose and pad by continuity*/
void
econvolve(double*       dst_pt, 
	  const double* src_pt,    int M, int N,
	  const double* filter_pt, int W)
{
  int i,j ;
  for(j = 0 ; j < N ; ++j) { 
    for(i = 0 ; i < M ; ++i) {
      double        acc = 0.0 ;
      double const* g = filter_pt ;
      double const* start = src_pt + (i-W) ;
      double const* stop  ;
      double        x ;
      
      /* beginning */
      stop = src_pt + max(0, i-W) ;
      x    = *stop ;
      while( start <= stop ) { acc += (*g++) * x ; start++ ; }
      
      /* middle */
      stop =  src_pt + min(M-1, i+W) ;
      while( start <  stop ) acc += (*g++) * (*start++) ;
      
      /* end */
      x  = *start ;
      stop = src_pt + (i+W) ;
      while( start <= stop ) { acc += (*g++) * x ; start++ ; } 
      
      /* save */
      *dst_pt = acc ; 
      dst_pt += N ;      
    }
    /* next column */
    src_pt += M ;
    dst_pt -= M*N - 1 ;
  }
}

void
mexFunction(int nout, mxArray *out[], 
            int nin, const mxArray *in[])
{
  int M,N,K,ndims ;
  int const *dims ;
  double* I_pt ;
  double* J_pt ;
  double s ;
  enum {I=0,S} ;
  enum {J=0} ;

  /* ------------------------------------------------------------------
  **                                                Check the arguments
  ** --------------------------------------------------------------- */ 
  if (nin != 2) {
    mexErrMsgTxt("Exactly two input arguments required.");
  } else if (nout > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }
  
  if (!mxIsDouble(in[I]) || 
      !mxIsDouble(in[S]) ) {
    mexErrMsgTxt("All arguments must be real.") ;
  }
  
  if(mxGetNumberOfDimensions(in[I]) > 3||
     mxGetNumberOfDimensions(in[S]) > 2) {
    mexErrMsgTxt("I must be a two dimensional array and S a scalar.") ;
  }
  
  if(max(mxGetM(in[S]),mxGetN(in[S])) > 1) {
    mexErrMsgTxt("S must be a scalar.\n") ;
  }

  ndims = mxGetNumberOfDimensions(in[I]) ;
  dims  = mxGetDimensions(in[I]) ;
  M = dims[0] ;
  N = dims[1] ;
  K = (ndims > 2) ? dims[2] : 1 ;

  out[J] = mxCreateNumericArray(ndims, dims, mxDOUBLE_CLASS, mxREAL) ;
  
  I_pt   = mxGetPr(in[I]) ;
  J_pt   = mxGetPr(out[J]) ;
  s      = *mxGetPr(in[S]) ;

  /* ------------------------------------------------------------------
  **                                                         Do the job
  ** --------------------------------------------------------------- */ 
  if(s > 0.01) {
    
    int W = (int) ceil(4*s) ;
    int j,k ;
    double* g0 = (double*) mxMalloc( (2*W+1)*sizeof(double) ) ;
    double* buffer = (double*) mxMalloc( M*N*sizeof(double) ) ;
    double acc=0.0 ;
    
    for(j = 0 ; j < 2*W+1 ; ++j) {
      g0[j] = exp(-0.5 * (j - W)*(j - W)/(s*s)) ;
      acc += g0[j] ;
    }
    for(j = 0 ; j < 2*W+1 ; ++j) {
      g0[j] /= acc ;
    }

    for(k = 0 ; k < K ; ++k) {
#if defined PAD_BY_CONTINUITY

    econvolve(buffer, I_pt, M, N, g0, W) ;
    econvolve(J_pt, buffer, N, M, g0, W) ;

#else
    
    convolve(buffer, I_pt, M, N, g0, W) ;
    convolve(J_pt, buffer, N, M, g0, W) ;

#endif
      I_pt += M*N ;
      J_pt += M*N ;
    }
  
    mxFree(buffer) ;
    mxFree(g0) ;
  } else {
    memcpy(J_pt, I_pt, sizeof(double)*M*N) ;   
  }
}

#include "heat.h"
#include <omp.h>

#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;

    // number of blocks
    nbx = omp_get_max_threads();
    nby = 1;

    // number of elements inside the block
    bx = sizex/nbx;
    by = sizey/nby;

    #pragma omp parallel for reduction(+: sum) private(diff)
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    utmp[i*sizey+j]= 0.25 * (u[ i*sizey + (j-1)] +  // left
                            u[i*sizey + (j+1)] +  // right
                            u[(i-1)*sizey + j] +  // top
                            u[(i+1)*sizey + j]);  // bottom
                    diff = utmp[i*sizey+j] - u[i*sizey + j];
                    sum += diff * diff; 
                }

    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    // Computing "Red" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
                            u[ i*sizey	+ (j+1) ]+  // right
                            u[ (i-1)*sizey	+ j     ]+  // top
                            u[ (i+1)*sizey	+ j     ]); // bottom
                    diff = unew - u[i*sizey+ j];
                    sum += diff * diff; 
                    u[i*sizey+j]=unew;
                }
    }

    // Computing "Black" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
                            u[ i*sizey	+ (j+1) ]+  // right
                            u[ (i-1)*sizey	+ j     ]+  // top
                            u[ (i+1)*sizey	+ j     ]); // bottom
                    diff = unew - u[i*sizey+ j];
                    sum += diff * diff; 
                    u[i*sizey+j]=unew;
                }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss_ordered (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx =  omp_get_max_threads();
    bx = sizex/nbx;
    nby = omp_get_max_threads(); 
    by = sizey/nby;

    #pragma omp parallel for collapse(2) ordered(2) reduction(+:sum) private(diff, unew)
    for (int ii=0; ii<nbx; ii++){
        for (int jj=0; jj<nby; jj++){
            #pragma omp ordered depend(sink: ii-1, jj) depend(sink: ii,jj-1) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++){
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    unew= 0.25 * (u[ i*sizey + (j-1) ]+  // left
                            u[i*sizey+(j+1) ]+  // right
                            u[(i-1)*sizey+ j]+  // top
                            u[(i+1)*sizey+ j]); // bottom
                    diff = unew - u[i*sizey+ j];
                    sum += diff * diff; 
                    u[i*sizey+j]=unew;
                }
            }
            #pragma omp ordered depend(source)
        }
    }
    return sum;
}


/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss_task (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = omp_get_max_threads(); 
    bx = sizex/nbx;
    nby = omp_get_max_threads(); 
    by = sizey/nby;

    int blocks[nbx][nby];

    #pragma omp parallel 
    #pragma omp single
    {
        for (int ii=0; ii<nbx; ii++){
            for (int jj=0; jj<nby; jj++) {
                #pragma omp task depend(in: blocks[ii-1][jj], blocks[ii][jj-1] ) depend(inout: blocks[ii][jj]) private(diff, unew) 
                {
                    double partial_sum = 0.0;
                    for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++){
                        for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
                                    u[ i*sizey	+ (j+1) ]+  // right
                                    u[ (i-1)*sizey	+ j     ]+  // top
                                    u[ (i+1)*sizey	+ j     ]); // bottom
                            diff = unew - u[i*sizey+ j];
                            partial_sum += diff * diff; 
                            u[i*sizey+j]=unew;
                        }
                    }
                    #pragma omp atomic
                    sum += partial_sum;
                }
            }
        }
     }

    return sum;
}


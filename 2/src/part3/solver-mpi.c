#include "heat.h"
#include <mpi.h>

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
  
    for (int i=1; i<sizex-1; i++) 
        for (int j=1; j<sizey-1; j++) {
        utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
                 u[ i*sizey     + (j+1) ]+  // right
                     u[ (i-1)*sizey + j     ]+  // top
                     u[ (i+1)*sizey + j     ]); // bottom
        diff = utmp[i*sizey+j] - u[i*sizey + j];
        sum += diff * diff; 
    }

    return sum;
}

double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nby, by, numprocs;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request r;

    by = sizex - 2;
    nby = (sizey-2) / by;
    numprocs = nby;

    for (int jj = 0; jj < nby ; jj++) {
        if (rank > 0){
            int offset = jj * by + 1;
            MPI_Recv(u + offset, by, MPI_DOUBLE, rank - 1, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }

        for (int i=1; i < sizex-1; i++) {
            for (int j=1+jj*by; j < (jj+1)*by + 1; j++) {
                unew= 0.25 * 
                      ( u[ i*sizey + (j-1) ]+  // left
                        u[ i*sizey + (j+1) ]+  // right
                        u[ (i-1) * sizey + j ]+  // top
                        u[ (i+1) * sizey + j ]); // bottom

                diff = unew - u[i*sizey+j];
                sum += diff * diff; 
                u[i*sizey+j]=unew;
            }
        }

        if (rank < numprocs - 1 ){
            int offset = (sizex - 2) * sizey +  jj * by + 1;
            MPI_Isend( u + offset, by, MPI_DOUBLE, rank + 1, 0, 
                    MPI_COMM_WORLD, &r);
        }
    }

    return sum;
}


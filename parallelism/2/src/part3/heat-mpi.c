#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"
#include <string.h>


void usage( char *s )
{
    fprintf(stderr, 
            "Usage: %s <input file> [result file]\n\n", s);
}

int main( int argc, char *argv[] )
{
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs-1);

        // algorithmic parameters
        algoparam_t param;
        int np;

        double runtime, flop;
        double residual=0.0;

        // check arguments
        if( argc < 2 )
        {
            usage( argv[0] );
            return 1;
        }

        // check input file
        if( !(infile=fopen(argv[1], "r"))  ) 
        {
            fprintf(stderr, 
                    "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);

            usage(argv[0]);
            return 1;
        }

        // check result file
        resfilename= (argc>=3) ? argv[2]:"heat.ppm";

        if( !(resfile=fopen(resfilename, "w")) )
        {
            fprintf(stderr, 
                    "\nError: Cannot open \"%s\" for writing.\n\n", 
                    resfilename);
            usage(argv[0]);
            return 1;
        }

        // check input
        if( !read_input(infile, &param) )
        {
            fprintf(stderr, "\nError: Error parsing input file.\n\n");
            usage(argv[0]);
            return 1;
        }
        print_params(&param);

        // set the visualization resolution

        param.u     = 0;
        param.uhelp = 0;
        param.uvis  = 0;
        param.visres = param.resolution;

        if( !initialize(&param) )
        {
            fprintf(stderr, "Error in Solver initialization.\n\n");
            usage(argv[0]);
            return 1;
        }

        // full size (param.resolution are only the inner points)
        np = param.resolution + 2;

        int rows = param.resolution/numprocs;

        // starting time
        runtime = wtime();

        MPI_Bcast(&param.maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&param.resolution, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&param.algorithm, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int send_sz = (rows + 2) * np;
        // send to workers the necessary data to perform computation
        for (int id=1; id<numprocs; id++) {
            int offset = rows * np * id;

            MPI_Send(param.u + offset, send_sz, MPI_DOUBLE, id, 0, MPI_COMM_WORLD);
            MPI_Send(param.uhelp + offset, send_sz, MPI_DOUBLE, id, 0, MPI_COMM_WORLD);
        }

        iter = 0;
        while(1) {
            switch( param.algorithm ) {
                case 0: // JACOBI
                    MPI_Recv(param.u+(rows+1)*np, np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(param.u+rows*np, np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD );

                    residual = relax_jacobi(param.u, param.uhelp, rows + 2, np);

                    memcpy(param.u, param.uhelp, sizeof(double)*(rows+2)*np);
                    break;
                case 1: // GAUSS
                    residual = relax_gauss(param.u, rows+2, np);
                    if (myid < numprocs - 1){
                        MPI_Recv(param.u+np*(rows+1), np, MPI_DOUBLE, myid+1, 0, 
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }

                    break;
            }

            iter++;
            MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            if (residual < 0.00005) break;
            if (param.maxiter>0 && iter>=param.maxiter) break;
        }

        int recv_sz = param.resolution/numprocs * np;
        
        for (int id = 1; id < numprocs; id++){
            int offset = param.resolution/numprocs * np * id + np;
            MPI_Recv(param.u + offset, recv_sz, 
                     MPI_DOUBLE, id, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Flop count after iter iterations
        flop = iter * 11.0 * param.resolution * param.resolution;
        // stopping time
        runtime = wtime() - runtime;

        fprintf(stdout, "Time: %04.3f ", runtime);
        fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
                flop/1000000000.0,
                flop/runtime/1000000);
        fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

        // for plot...
        coarsen( param.u, np, np,
                param.uvis, param.visres+2, param.visres+2 );

        write_image( resfile, param.uvis,  
                param.visres+2, 
                param.visres+2 );

        finalize( &param );

        fprintf(stdout, "Process %d finished computing with residual value = %f\n", myid, residual);
        MPI_Finalize();

        return 0;
    } else {

        printf("I am worker %d and ready to receive work to do ...\n", myid);

        // receive information from master to perform computation locally
        int columns, rows, np;
        int iter, maxiter;
        int algorithm;
        double residual;

        MPI_Bcast(&maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&algorithm, 1, MPI_INT, 0, MPI_COMM_WORLD);

        np = columns + 2;
        rows = columns/numprocs;

        // allocate memory for worker
        double * u = calloc( sizeof(double),(rows+2)*(columns+2) );
        double * uhelp = calloc( sizeof(double),(rows+2)*(columns+2) );
        if( (!u) || (!uhelp) )
        {
            fprintf(stderr, "Error: Cannot allocate memory\n");
            return 0;
        }

        // fill initial values for matrix with values received from master
        MPI_Recv(u, (rows+2)*(columns+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(uhelp, (rows+2)*(columns+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

        iter = 0;
        while(1) {
            switch( algorithm ) {
                case 0: // JACOBI
                    MPI_Send(u+np, np, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD );
                    MPI_Recv(u, np, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (myid < numprocs - 1){
                        MPI_Send(u+rows*np, np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD );
                        MPI_Recv(u+(rows+1)*np, np, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }

                    residual = relax_jacobi(u, uhelp, rows+2, columns+2);
                    memcpy(u, uhelp, sizeof(double)*(rows+2)*(columns+2));
                    break;
                case 1: // GAUSS
                    residual = relax_gauss(u, rows+2, columns+2);

                    MPI_Send(u+np, np, MPI_DOUBLE, myid-1, 0, 
                            MPI_COMM_WORLD);
                    if (myid < numprocs - 1){
                        MPI_Recv(u+np*(rows+1), np, MPI_DOUBLE, myid+1, 0, 
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    break;
            }
            iter++;
            MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            if (residual < 0.00005) break;
            if (maxiter>0 && iter>=maxiter) break;
        }

        MPI_Send(u+np, rows*np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        if( u ) free(u); 
        if( uhelp ) free(uhelp);

        fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", myid, iter, residual);

        MPI_Finalize();
        exit(0);
    }
}

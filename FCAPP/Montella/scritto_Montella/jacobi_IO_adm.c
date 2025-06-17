/**
 *
 *    Jacobi iterative method
 *
 */
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <empi.h>

void generate_matrix (int dim, double *A, double *B);

int main (int argc, char *argv[])
{
    double *matrixA = NULL;
    double *vectorB = NULL;
    double *vectorX_local = NULL;
    double *vectorX_global = NULL;
    int dim = 6000;
    int ldim = 0;
    int it=0;
    int itmax=1000;

    // book memory for arrays
    matrixA = (double*) calloc (dim*dim, sizeof (double));
    vectorB = (double*) calloc (dim, sizeof (double));
    vectorX_local = (double*) calloc (dim, sizeof (double));
    vectorX_global = (double*) calloc (dim, sizeof (double));

    // init MPI and get world_rank and world_size
    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(ADM_COMM_WORLD, &world_rank);
    MPI_Comm_size(ADM_COMM_WORLD, &world_size);

    // get processor name
    int len;
    char mpi_name[128];
    MPI_Get_processor_name(mpi_name, &len);

    // compute local dimension for this process
    ldim=dim/world_size;

    printf ("Rank(%d/%d): global dimension=%d, local_dimension=%d\n", world_rank, world_size, dim, ldim);

    // Generates the matrix
    if (world_rank == 0) {
        generate_matrix (dim, matrixA, vectorB);
    }

    //Bcast matrixes
    MPI_Bcast (matrixA, dim*dim, MPI_DOUBLE, 0, ADM_COMM_WORLD);
    MPI_Bcast (vectorB, dim, MPI_DOUBLE, 0, ADM_COMM_WORLD);

//--

    // get process type
    int proctype;
    ADM_GetSysAttributesInt ("ADM_GLOBAL_PROCESS_TYPE", &proctype);

    // if process is native
    if (proctype == ADM_NATIVE) {
        printf ("Rank(%d/%d): Process native\n", world_rank, world_size);
    // if process is spawned
    } else {
        //Bcast result vector
        MPI_Bcast (vectorX_global, dim, MPI_DOUBLE, 0, ADM_COMM_WORLD);

        printf ("Rank(%d/%d): Process spawned\n", world_rank, world_size);
    }

    /* set max number of iterations */
    ADM_RegisterSysAttributesInt ("ADM_GLOBAL_MAX_ITERATION", &itmax);

    /* get actual iteration for new added processes*/
    ADM_GetSysAttributesInt ("ADM_GLOBAL_ITERATION", &it);

    /* starting monitoring service */
    ADM_MonitoringService (ADM_SERVICE_START);

    // init last world zize
    int last_world_size = world_size;

    // start loop
    for (; it < itmax; it ++) {

        //Select hints on specific iterations
        int procs_hint = 0;
        int excl_nodes_hint = 0;
        if ( (it == 2*(itmax/10)) || (it == 4*(itmax/10)) ){
            procs_hint = 2;
            excl_nodes_hint = 0;
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_NUM_PROCESS", &procs_hint);
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_EXCL_NODES", &excl_nodes_hint);
            printf ("Rank(%d/%d): Iteration= %d, procs_hint=%d, excl_nodes_hint=%d\n", world_rank, world_size, it, procs_hint, excl_nodes_hint);
        } else if ( (it == 6*(itmax/10)) || (it == 8*(itmax/10)) ){
            procs_hint = -2;
            excl_nodes_hint = 0;
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_NUM_PROCESS", &procs_hint);
            ADM_RegisterSysAttributesInt ("ADM_GLOBAL_HINT_EXCL_NODES", &excl_nodes_hint);
            printf ("Rank(%d/%d): Iteration= %d, procs_hint=%d, excl_nodes_hint=%d\n", world_rank, world_size, it, procs_hint, excl_nodes_hint);
        }

        // update matrices if new spawned processes
        if (last_world_size != world_size) {
            if (last_world_size < world_size) {
                //Bcast matrixes
                MPI_Bcast (matrixA, dim*dim, MPI_DOUBLE, 0, ADM_COMM_WORLD);
                MPI_Bcast (vectorB, dim, MPI_DOUBLE, 0, ADM_COMM_WORLD);

                //Bcast result vector
                MPI_Bcast (vectorX_global, dim, MPI_DOUBLE, 0, ADM_COMM_WORLD);         //attenzione a questo

            }
            last_world_size = world_size;
            ldim=dim/world_size;
        }

        /* start malelability region */
        ADM_MalleableRegion (ADM_SERVICE_START);

        printf ("Rank(%d/%d): Iteration= %d, global dimension=%d, local_dimension=%d\n", world_rank, world_size, it, dim, ldim);

        //perform the operations
        for (int i=0; i<ldim; i++) {
            int i_global = i+world_rank*ldim;
            double aux = 0.0;
            for (int j=0; j<dim; j++) {
                if (j!=i_global) {
                    aux = aux + matrixA[(i*dim)+j] * vectorX_global[j];
                }
            }
            vectorX_local[i] = (vectorB[i_global]-aux)/matrixA[(i*dim)+i_global];

        }

        // gather vector X
        MPI_Allgather (vectorX_local, ldim, MPI_DOUBLE, vectorX_global, ldim,  MPI_DOUBLE, ADM_COMM_WORLD);

        // update the iteration value
        ADM_RegisterSysAttributesInt ("ADM_GLOBAL_ITERATION", &it);

        // ending malleable region
        int status;
        status = ADM_MalleableRegion (ADM_SERVICE_STOP);

        // check if process ended after malleable region
        if (status == ADM_ACTIVE) {
            // updata world_rank and size
            MPI_Comm_rank(ADM_COMM_WORLD, &world_rank);
            MPI_Comm_size(ADM_COMM_WORLD, &world_size);
        } else {
            // end the process
            break;
        }
    }

    /* ending monitoring service */
    ADM_MonitoringService (ADM_SERVICE_STOP);

    printf("Rank (%d/%d): End of loop \n", world_rank, world_size);

    MPI_Finalize();

    printf("Rank (%d/%d): End of process %d\n", world_rank, world_size, world_rank);
    if (world_rank == 0) {
        printf("Rank (%d/%d): End of Application\n", world_rank, world_size);
    }
    free(matrixA);
    free(vectorB);
    free(vectorX_local);
    free(vectorX_global);

    return 0;
}


// Generate matrix's values
void generate_matrix (int dim, double *A, double *B) {

    for (int i = 0; i < dim; i ++) {

        //printf ("B: i=%d\n",i);
        B[i] = (i % 5) + 1;

        for (int j = 0; j < dim; j ++){

            //A[(i*dim)+j]= 30-60*(rand() / RAND_MAX );  // Original random distribution
            //printf ("A: i=%d, j=%d, (i*dim)+j=%d\n",i,j,(i*dim)+j);

            A[(i*dim)+j]=(double)((i*dim)+j); // For I/O debugging purposes
        }
    }

    for (int i = 0; i < dim; i ++) {
        double temp = 0.0;
        for (int j = 0; j < dim; j ++){
            //printf ("A(AGAIN): i=%d, j=%d,  (i*dim)+j=%d\n",i,j,(i*dim)+j);
            temp = temp + A[(i*dim)+j];
        }
        //printf ("A(----): i=%d, (i*dim)+i=%d\n",i, (i*dim)+i);
        A[(i*dim)+i] = temp + dim;
    }
}
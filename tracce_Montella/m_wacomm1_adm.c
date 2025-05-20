#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <empi.h>

struct particle_data {
    unsigned long id;
    double k;
    double j;
    double i;
    double health;
    double age;
    double time;
};


int main(int argc, char *argv[]){
    unsigned long int particleId = 0;
    int newParticlesPerIt = 6000;
    int nParticles = 0;
    int idx;
    struct particle_data *particles = NULL;

    char mpi_name[128];
    int len, it = 0, itmax =168;

    char bin[1024];
    sprintf(bin,"%s",argv[0]);

    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(ADM_COMM_WORLD, &world_rank);
    MPI_Comm_size(ADM_COMM_WORLD, &world_size);

    MPI_Get_processor_name (mpi_name, &len);

    if (world_rank == 0) {
        nParticles=0;
    }

    // get process type
    int proctype;
    ADM_GetSysAttributesInt ("ADM_GLOBAL_PROCESS_TYPE", &proctype);

    // if process is native
    if (proctype == ADM_NATIVE) {
        printf ("Rank(%d/%d): Process native\n", world_rank, world_size);
    // if process is spawned
    } else {
        printf ("Rank(%d/%d): Process spawned\n", world_rank, world_size);
    }

    /* set max number of iterations */
    ADM_RegisterSysAttributesInt ("ADM_GLOBAL_MAX_ITERATION", &itmax);

    /* get actual iteration for new added processes*/
    ADM_GetSysAttributesInt ("ADM_GLOBAL_ITERATION", &it);

    /* starting monitoring service */
    ADM_MonitoringService (ADM_SERVICE_START);

    while (it < itmax) {

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
        /* start malelability region */
        ADM_MalleableRegion (ADM_SERVICE_START);

        int *send_counts = (int *) malloc(world_size * sizeof(int));

        int *displs = (int *) malloc(world_size * sizeof(int));

        struct particle_data *sendbuf = NULL;

        struct particle_data *recvbuf  = NULL;
        MPI_Barrier (ADM_COMM_WORLD);

        struct particle_data *localParticles  = NULL;

        if (world_rank == 0) {
            // Emit particles
            int nParticles0 = nParticles;


            struct particle_data *particles0=particles;
            if (it <= itmax/2) {
              nParticles += newParticlesPerIt;
            } else {
              nParticles -= newParticlesPerIt;
            }
            particles = (struct particle_data *)malloc(sizeof(struct particle_data) * nParticles);
            if (particles0) {
                memcpy(particles,particles0,nParticles0);
            }
            int particleIdx = 0;
            for(particleIdx = nParticles0; particleIdx<nParticles; particleIdx++) {
                particles[particleIdx].id = particleId;
                particles[particleIdx].i = 0;
                particles[particleIdx].j = 0;
                particles[particleIdx].k = 0;
                particles[particleIdx].health = 1;
                particles[particleIdx].age = 0;
                particles[particleIdx].time = 0;
                particleId++;
            }
            if (particles0 && it != 0) {
                free(particles0);
            }

            printf("\nRank (0/%d): Iteration %d, Total number of particles: %d\n", world_size, it, nParticles);

            // Get the number of particles to be processed for each process
            int  particlesPerProcess = nParticles / world_size;

            // Get the number of spare particles for the process with world_rank==0
                        int spare = nParticles % world_size;

            // The process with world_rank==0 will calculate extra particles
            send_counts[0] = (int)((particlesPerProcess + spare));

            // The process with world_rank==0 will process the first particles
                        displs[0]=0;

            // Prepare counts and displacement for data distribution
            // For each process...
            int i;
            for (i = 1; i < world_size; i++) {

                // Set the number of particles per process
                send_counts[i] = (int)(particlesPerProcess);

                                // Set the displacement if terms of particle_data size
                displs[i]=send_counts[0]+particlesPerProcess*(i-1);
            }

            // Prepare a buffer of particle_data
            sendbuf=(struct particle_data *)malloc(nParticles*sizeof(struct particle_data));

            // For each particle...
            for (idx=0;idx<nParticles;idx++) {

                // Copy data into the buffer
                sendbuf[idx]=particles[idx];
            }
        }


        // Broadcast the number of particles for each processor
        MPI_Bcast(send_counts,world_size,MPI_INT,0,ADM_COMM_WORLD);

        // Get the numeber of particles to process for the current processor
        int particlesToProcess=send_counts[world_rank];

        printf("Rank (%d/%d): Iteration %d, number of particles for process %d: %d\n", world_rank, world_size, it, world_rank, particlesToProcess);

        // Allocate the receiving buffer
        recvbuf=(struct particle_data *)malloc(particlesToProcess * sizeof(struct particle_data));

        // Define a variable that will contain the mpiError
        int mpiError;

        // Define a MPI struct miming particle_data struct

        // Set the number of fields
        int num_members = 7;

        // Set the cardinality of each field
        int lengths[] = { 1, 1, 1, 1, 1, 1, 1 };

        // Define an array of MPI int containing the offset of each struct field
        MPI_Aint offsets[] = {
            offsetof(struct particle_data, id),
                offsetof(struct particle_data, k),
                offsetof(struct particle_data, j),
                offsetof(struct particle_data, i),
                offsetof(struct particle_data, health),
                offsetof(struct particle_data, age),
                offsetof(struct particle_data, time)
        };

        // Define an array of MPI data type containing the MPI type of each field
        MPI_Datatype types[] = {
                MPI_UNSIGNED_LONG,
                MPI_DOUBLE,
                MPI_DOUBLE,
            MPI_DOUBLE,
            MPI_DOUBLE,
            MPI_DOUBLE,
            MPI_DOUBLE
        };

        // Define a container for the new MPI data type
        MPI_Datatype mpiParticleData;

        // Create the MPI struct
        MPI_Type_create_struct(num_members, lengths, offsets, types, &mpiParticleData);

        // Add the new datatype
        MPI_Type_commit(&mpiParticleData);

        // Distribute to all processes the send buffer
        mpiError=MPI_Scatterv(sendbuf, send_counts, displs, mpiParticleData,
            recvbuf, particlesToProcess, mpiParticleData, 0, ADM_COMM_WORLD);
        if (mpiError != MPI_SUCCESS) printf("Rank(%d/%d): MPI_Scatterv: error", world_rank, world_size);
        localParticles = (struct particle_data *)malloc(particlesToProcess*sizeof(struct particle_data));
        memcpy(localParticles, recvbuf,particlesToProcess);

        // move...
        for(idx = 0;idx < particlesToProcess;idx++) {
            int t;
            for (t = 0; t < 120; t++) {
                localParticles[idx].i = localParticles[idx].i + (2.0*rand()/RAND_MAX)-1;
                localParticles[idx].j = localParticles[idx].j + (2.0*rand()/RAND_MAX)-1;
                localParticles[idx].k = localParticles[idx].k + (2.0*rand()/RAND_MAX)-1;
            }
        }


        memcpy(recvbuf,localParticles,particlesToProcess);

        // Send the receiving buffer to the process with world_rank==0
        MPI_Gatherv(recvbuf, send_counts[world_rank],mpiParticleData,
            sendbuf,send_counts,displs,mpiParticleData,0,ADM_COMM_WORLD);



        // Remove the MPI Data type
        MPI_Type_free(&mpiParticleData);

        free(localParticles);

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

        // increment iteration
        it++;
    }

    /* ending monitoring service */
    ADM_MonitoringService (ADM_SERVICE_STOP);

    printf("Rank (%d/%d): End of loop \n", world_rank, world_size);

    MPI_Finalize();

    printf("Rank (%d/%d): End of process %d\n", world_rank, world_size, world_rank);
    if (world_rank == 0) {
        printf("Rank (%d/%d): End of Application\n", world_rank, world_size);
    }
    return 0;
}
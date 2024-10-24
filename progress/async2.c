#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT (1024*1024*1024)

int main(void)
{
    MPI_Request req;
    int rank, size;
    void *buf;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    buf = malloc(COUNT);
    
    if (rank == 0) {
        MPI_Barrier(MPI_COMM_WORLD);
	MPI_Isend(buf, COUNT, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &req);
    } else if (rank == 1) {
	MPI_Irecv(buf, COUNT, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    sleep(2);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    MPI_Finalize();
    return 0;
}

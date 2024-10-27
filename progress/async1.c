#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT (1024*1024*1024)

int main(void)
{
    double start, end;
    MPI_Request req;
    int rank, size;
    void *buf;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    buf = malloc(COUNT);
    
    if (rank == 0) {
	MPI_Isend(buf, COUNT, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &req);
#ifdef SLEEP
        usleep(100000);
#elif defined(TEST)
        int flag;
        usleep(50000);
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
        usleep(50000);
#endif
    } else if (rank == 1) {
	MPI_Irecv(buf, COUNT, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
    }

    start = MPI_Wtime();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    end = MPI_Wtime();
    if (rank == 0) {
        printf("send wait time %f\n", end - start);
    }
    if (rank == 1) {
        printf("recv wait time %f\n", end - start);
    }

    MPI_Finalize();
    return 0;
}

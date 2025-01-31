#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define NUM_ITERS 1000
#define NUM_DOUBLES 16

static int wrank;
static int wsize;
static int *get_neighbors(int *num_neighbors);

int main(void)
{
    int rank, size;
    double start, end;
    double total_time = 0;
    double avg_time;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    if (wsize < 10) {
        fprintf(stderr, "run with at least 10 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    srand(rank); /* setup for rand() usage */

    /* send and recv bufs used for both implementations */
    double *sendbuf = malloc(sizeof(double) * NUM_DOUBLES * wsize);
    double *recvbuf = malloc(sizeof(double) * NUM_DOUBLES * wsize);

    /* alltoallv implementation */
    int *sendcounts = malloc(sizeof(int) * wsize);
    int *recvcounts = malloc(sizeof(int) * wsize);
    int *displs = malloc(sizeof(int) * wsize);
    for (int i = 0; i < wsize; i++) {
        displs[i] = i * NUM_DOUBLES;
    }
    for (int i = 0; i < NUM_ITERS; i++) {
        int num_neighbors;
        int *neighbors = get_neighbors(&num_neighbors);

        start = MPI_Wtime();
        memset(sendcounts, 0, sizeof(int) * wsize);
        for (int j = 0; j < num_neighbors; j++) {
            sendcounts[neighbors[j]] = NUM_DOUBLES;
        }
        MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Alltoallv(sendbuf, sendcounts, displs, MPI_DOUBLE, recvbuf, recvcounts,
                      displs, MPI_DOUBLE, MPI_COMM_WORLD);
        end = MPI_Wtime();
        total_time += end - start;
    }
    MPI_Reduce(&total_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (wrank == 0) {
        printf("avg alltoallv time = %f\n", avg_time / wsize);
    }
    free(sendcounts);
    free(recvcounts);
    free(displs);

    /* ibarrier implementation */
    total_time = 0;
    MPI_Request *sreqs = malloc(sizeof(MPI_Request) * wsize);
    for (int i = 0; i < NUM_ITERS; i++) {
        /* determine neighbors for this iteration */
        int num_neighbors;
        int *neighbors = get_neighbors(&num_neighbors);

        start = MPI_Wtime();
        for (int j = 0; j < num_neighbors; j++) {
            int dest = neighbors[j];
            double *buf = sendbuf + (dest * NUM_DOUBLES);

            MPI_Issend(buf, 16, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &sreqs[j]);
        }

        MPI_Request barrier_req = MPI_REQUEST_NULL;
        while (1) {
            MPI_Status status;
            int flag;

            /* check for incoming messages */
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                int src = status.MPI_SOURCE;
                double *buf = recvbuf + (src * NUM_DOUBLES);

                MPI_Recv(buf, NUM_DOUBLES, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (barrier_req != MPI_REQUEST_NULL) {
                MPI_Test(&barrier_req, &flag, MPI_STATUS_IGNORE);
                if (flag) {
                    /* barrier completed, this iteration is done */
                    free(neighbors);
                    end = MPI_Wtime();
                    total_time += end - start;
                    break;
                }
            } else {
                MPI_Testall(num_neighbors, sreqs, &flag, MPI_STATUSES_IGNORE);
                if (flag) {
                    /* sends are done, start barrier */
                    MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
                }
            }
        }
    }
    free(sreqs);
    MPI_Reduce(&total_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (wrank == 0) {
        printf("avg ibarrier time = %f\n", avg_time / wsize);
    }

    MPI_Finalize();
    return 0;
}

static int *get_neighbors(int *num_neighbors)
{
    int n = wsize / 10;
    int *neighbors = malloc(sizeof(int) * n);

    for (int i = 0; i < n; i++) {
        int neighbor;
        do {
            neighbor = rand() % wsize;
        } while (neighbor == wrank);
        neighbors[i] = neighbor;
    }

    *num_neighbors = n;
    return neighbors;
}

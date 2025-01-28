#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MESSAGE_SIZE 8
#define NUM_ITER 50
#define WINDOW_SIZE 64
#define NUM_MESSAGES (NUM_ITER * WINDOW_SIZE)
#define NUM_THREADS 2

MPI_Comm t_comms[NUM_THREADS];
double t_elapsed[NUM_THREADS];
void *t_bufs[NUM_THREADS];
void do_msg_rate(int rank, int n);

int main(void)
{
    double msg_rate;
    int rank, size;
    int provided;
    /* Alignment prevents either false-sharing on the CPU or serialization in the
     * NIC's parallel TLB engine. Use page size to be safe. */
    int buffer_align = sysconf(_SC_PAGESIZE);

    /* setup MPI */
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI_THREAD_MULTIPLE required for this test.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        fprintf(stderr, "run with exactly two processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* allocate per thread resources */
    for (int i = 0; i < NUM_THREADS; i++) {
        MPI_Comm_dup(MPI_COMM_WORLD, &t_comms[i]);
        posix_memalign(&t_bufs[i], buffer_align, MESSAGE_SIZE);
    }

    /* run with 1 thread */
    do_msg_rate(rank, 1);
    msg_rate = ((double) NUM_MESSAGES / t_elapsed[0]) / 1e6;

    /* run with multiple threads */
    do_msg_rate(rank, NUM_THREADS);

    /* calculate message rate */
    if (rank == 0) {
        double mt_msg_rate;
        printf("Number of messages: %d\n", NUM_MESSAGES);
        printf("Message size: %d\n", MESSAGE_SIZE);
        printf("Window size: %d\n", WINDOW_SIZE);
        printf("Mmsgs/s with one thread: %-10.2f\n\n", msg_rate);
        printf("%-10s\t%-10s\n", "Thread", "Mmsgs/s");

        mt_msg_rate = 0;
        for (int tid = 0; tid < NUM_THREADS; tid++) {
            double my_msg_rate = ((double) NUM_MESSAGES / t_elapsed[tid]) / 1e6;
            printf("%-10d\t%-10.2f\n", tid, my_msg_rate);
            mt_msg_rate += my_msg_rate;
        }
        printf("\n%-10s\t%-10s\t%-10s\n", "Size", "Threads", "Mmsgs/s");
        printf("%-10d\t%-10d\t%-10.2f\n", MESSAGE_SIZE, NUM_THREADS,
               mt_msg_rate);
    }

    /* free resources */
    for (int i = 0; i < NUM_THREADS; i++) {
        MPI_Comm_free(&t_comms[i]);
        free(t_bufs[i]);
    }

    MPI_Finalize();
    return 0;
}

void do_msg_rate(int rank, int n)
{
#pragma omp parallel num_threads(n)
    {
        int tid = omp_get_thread_num();
#ifdef SINGLECOMM
        MPI_Comm comm = MPI_COMM_WORLD;
#else
        MPI_Comm comm = t_comms[tid];
#endif
        void *buf = t_bufs[tid];
        MPI_Request requests[WINDOW_SIZE];
        double t_start, t_end;

        if (rank == 0) {
            t_start = MPI_Wtime();
        }

        for (int i = 0; i < NUM_ITER; i++) {
            for (int j = 0; j < WINDOW_SIZE; j++) {
                if (rank == 0) {
                    MPI_Isend(buf, MESSAGE_SIZE, MPI_BYTE, 1, 0, comm, &requests[j]);
                } else {
                    MPI_Irecv(buf, MESSAGE_SIZE, MPI_BYTE, 0, 0, comm, &requests[j]);
                }
            }
            MPI_Waitall(WINDOW_SIZE, requests, MPI_STATUSES_IGNORE);
        }

        if (rank == 0) {
            t_end = MPI_Wtime();
            t_elapsed[tid] = t_end - t_start;
        }
    }
}

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

/* Alignment prevents either false-sharing on the CPU or serialization in the
 * NIC's parallel TLB engine. Using page size for optimum results. */
#define BUFFER_ALIGNMENT 4096

#define MESSAGE_SIZE 8
#define NUM_MESSAGES 64000
#define WINDOW_SIZE 64
#define NUM_THREADS 2

MPI_Comm t_comms[NUM_THREADS];
double t_elapsed[NUM_THREADS];
void *t_bufs[NUM_THREADS];

static void msgrate_test(int rank, int tid);

int main(void)
{
    int rank, size;
    int provided;
    double msg_rate;

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
        posix_memalign(&t_bufs[i], BUFFER_ALIGNMENT, MESSAGE_SIZE);
    }

    /* Run test with 1 thread */
    msgrate_test(rank, 0);
    msg_rate = ((double) NUM_MESSAGES / t_elapsed[0]) / 1e6;

    /* Run test with multiple threads */
#pragma omp parallel num_threads(NUM_THREADS)
    {
        msgrate_test(rank, omp_get_thread_num());
    }

    /* Calculate message rate with multiple threads */
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

    for (int i = 0; i < NUM_THREADS; i++) {
        MPI_Comm_free(&t_comms[i]);
        free(t_bufs[i]);
    }

    MPI_Finalize();
    return 0;
}

static void msgrate_test(int rank, int tid)
{
    MPI_Comm my_comm = t_comms[tid];
    void *buf = t_bufs[tid];
    int win_i, win_post_i, win_posts;
    MPI_Request requests[WINDOW_SIZE];
    double t_start, t_end;

    win_posts = NUM_MESSAGES / WINDOW_SIZE;
    assert(win_posts * WINDOW_SIZE == NUM_MESSAGES);

    /* Benchmark */
    if (rank == 0) {
        t_start = MPI_Wtime();
    }

    for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
        for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
            if (rank == 0) {
                MPI_Isend(buf, MESSAGE_SIZE, MPI_CHAR, 1, tid, my_comm, &requests[win_i]);
            } else {
                MPI_Irecv(buf, MESSAGE_SIZE, MPI_CHAR, 0, tid, my_comm, &requests[win_i]);
            }
        }
        MPI_Waitall(WINDOW_SIZE, requests, MPI_STATUSES_IGNORE);
    }

    if (rank == 0) {
        t_end = MPI_Wtime();
        t_elapsed[tid] = t_end - t_start;
    }
}

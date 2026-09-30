#define _GNU_SOURCE
// ex07_pi_any_bsend.c - Exercise 7
// Exercise 6 (MPI_ANY_SOURCE receive) with the workers using Buffered Send.
//
// Each worker attaches its own buffer, copies its hit count into it with
// MPI_Bsend and returns immediately - it never waits for rank 0 to post the
// matching receive. MPI_Buffer_detach before MPI_Finalize guarantees the
// message has left before the process tears down.
//
// Build: mpicc -O2 ex07_pi_any_bsend.c -o ex07_pi_any_bsend
// Run  : mpirun -np 4 ./ex07_pi_any_bsend

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long darts = (argc > 1) ? atoll(argv[1]) : 10000000LL;

    long long base      = darts / size;
    long long remainder = darts % size;
    long long my_darts  = base + (rank < remainder ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    unsigned int seed = 12345u + 7919u * (unsigned int)rank;

    long long local_hits = 0;
    for (long long i = 0; i < my_darts; i++) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX;
        double y = (double)rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0) local_hits++;
    }

    long long total_hits = 0;
    if (rank == 0) {
        total_hits = local_hits;
        long long recv_hits = 0;
        MPI_Status status;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&recv_hits, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 0,
                     MPI_COMM_WORLD, &status);
            total_hits += recv_hits;
            printf("rank 0: got %lld hits from rank %d (buffered send)\n",
                   recv_hits, status.MPI_SOURCE);
            fflush(stdout);
        }
    } else {
        // One outstanding message of one long long per worker.
        int buffer_size = sizeof(long long) + MPI_BSEND_OVERHEAD;
        char *buffer = malloc(buffer_size);
        MPI_Buffer_attach(buffer, buffer_size);

        MPI_Bsend(&local_hits, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        // local_hits could safely be overwritten right here: Bsend already
        // copied it into the attached buffer.

        MPI_Buffer_detach(&buffer, &buffer_size);   // blocks until delivered
        free(buffer);
    }

    double elapsed = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("rank %d of %d : darts = %lld, hits = %lld (%.6f s)\n",
           rank, size, my_darts, local_hits, elapsed);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double pi = 4.0 * (double)total_hits / (double)darts;
        printf("\n[MPI_ANY_SOURCE + MPI_Bsend version]\n");
        printf("darts        = %lld\n", darts);
        printf("processes    = %d\n", size);
        printf("hits         = %lld\n", total_hits);
        printf("pi estimate  = %.10f\n", pi);
        printf("error        = %.10f\n", fabs(pi - M_PI));
        printf("time         = %.6f s\n", max_elapsed);
        printf("CSV,%d,%.6f\n", size, max_elapsed);
    }

    MPI_Finalize();
    return 0;
}

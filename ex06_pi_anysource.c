#define _GNU_SOURCE
// ex06_pi_anysource.c - Exercise 6
// Exercise 3 with the receive changed to MPI_ANY_SOURCE.
//
// Note on the lab wording: MPI_ANY_SOURCE is the wildcard for the SOURCE
// argument of MPI_Recv, not for the tag (the tag wildcard is MPI_ANY_TAG).
// So the change is  MPI_Recv(..., src, 0, ...)  ->  MPI_Recv(..., MPI_ANY_SOURCE, 0, ...).
// MPI_Status is now needed to find out which rank the message actually came from.
//
// Build: mpicc -O2 ex06_pi_anysource.c -o ex06_pi_any
// Run  : mpirun -np 4 ./ex06_pi_any

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
            // Take whichever worker finishes first, in arrival order.
            MPI_Recv(&recv_hits, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 0,
                     MPI_COMM_WORLD, &status);
            total_hits += recv_hits;
            printf("rank 0: got %lld hits from rank %d (tag %d)\n",
                   recv_hits, status.MPI_SOURCE, status.MPI_TAG);
            fflush(stdout);
        }
    } else {
        MPI_Send(&local_hits, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
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
        printf("\n[MPI_ANY_SOURCE version]\n");
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

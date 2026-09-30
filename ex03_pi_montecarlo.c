#define _GNU_SOURCE
// ex03_pi_montecarlo.c - Exercise 3
// Monte Carlo estimation of Pi over 10,000,000 samples, split across MPI ranks.
//
// Method (Dartmouth sample): throw random darts into the unit square, count how
// many land inside the quarter circle x^2 + y^2 <= 1.
//   area of quarter circle / area of square = (pi/4) / 1  =>  pi ~= 4 * hits / darts
//
// Each rank does darts/size throws with its OWN seed (rand_r, thread/process
// safe) and sends its hit count to rank 0 with MPI_Send. Rank 0 receives from
// every rank in rank order, which is the "original" version that Exercise 6
// will change to MPI_ANY_SOURCE.
//
// Build: mpicc -O2 ex03_pi_montecarlo.c -o ex03_pi
// Run  : mpirun -np 4 ./ex03_pi

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

    // Split the darts; first (darts % size) ranks take one extra.
    long long base      = darts / size;
    long long remainder = darts % size;
    long long my_darts  = base + (rank < remainder ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Distinct seed per rank, otherwise every rank repeats the same sequence
    // and the estimate is no better than a single rank's.
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
        for (int src = 1; src < size; src++) {
            // Fixed source order: rank 0 waits for rank 1 first, then 2, ...
            MPI_Recv(&recv_hits, 1, MPI_LONG_LONG, src, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_hits += recv_hits;
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
        printf("\ndarts        = %lld\n", darts);
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

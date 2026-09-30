// ex02_sum.c - Exercise 2
// Parallel sum of 1 .. 10,000,000 over multiple MPI processes / nodes.
// Collection is done with explicit point-to-point calls (MPI_Send / MPI_Recv),
// which is what the Part-1 lecture covers. MPI_Reduce is shown in a comment.
//
// Build: mpicc -O2 ex02_sum.c -o ex02_sum
// Run  : mpirun -np 4 ./ex02_sum
//        mpirun -np 8 --oversubscribe ./ex02_sum        (more ranks than cores)
//        mpirun -np 4 --hostfile hosts ./ex02_sum        (multiple nodes)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // N can be overridden on the command line, default is the lab value.
    long long N = (argc > 1) ? atoll(argv[1]) : 10000000LL;

    // Block decomposition: rank r owns [first, last].
    // The first (N % size) ranks take one extra element so nothing is lost
    // when N is not divisible by the number of processes.
    long long base      = N / size;
    long long remainder = N % size;
    long long first = rank * base + (rank < remainder ? rank : remainder) + 1;
    long long count = base + (rank < remainder ? 1 : 0);
    long long last  = first + count - 1;

    MPI_Barrier(MPI_COMM_WORLD);          // start all ranks from the same point
    double t0 = MPI_Wtime();

    long long local_sum = 0;
    for (long long i = first; i <= last; i++) local_sum += i;

    long long total = 0;
    if (rank == 0) {
        total = local_sum;                // rank 0's own share
        long long recv_sum = 0;
        for (int src = 1; src < size; src++) {
            MPI_Recv(&recv_sum, 1, MPI_LONG_LONG, src, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += recv_sum;
        }
    } else {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    // Equivalent one-liner (collective version):
    // MPI_Reduce(&local_sum, &total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;
    double max_elapsed = 0.0;
    // Wall time of the slowest rank is the real runtime of the job.
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("rank %d of %d : range [%lld, %lld] local_sum = %lld (%.6f s)\n",
           rank, size, first, last, local_sum, elapsed);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        long long expected = N * (N + 1) / 2;
        printf("\nN            = %lld\n", N);
        printf("processes    = %d\n", size);
        printf("parallel sum = %lld\n", total);
        printf("expected sum = %lld  (%s)\n", expected,
               total == expected ? "MATCH" : "MISMATCH");
        printf("time         = %.6f s\n", max_elapsed);
        printf("CSV,%d,%.6f\n", size, max_elapsed);   // parsed by run_scaling.sh
    }

    MPI_Finalize();
    return 0;
}

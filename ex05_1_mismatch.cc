// ex05_1_mismatch.cc - Exercise 5 part 1
// message1.cc modified so the send destination and the receive source do not
// match. Three modes, chosen by the command line argument:
//
//   mode 1 (default) : rank 0 sends to rank 1, but rank 1 waits for a message
//                      from rank 1. Nothing matches -> the program HANGS.
//   mode 2           : source matches but the tags do not -> also HANGS.
//   mode 3           : rank 0 sends to a destination rank that does not exist
//                      -> MPI reports MPI_ERR_RANK and aborts immediately.
//
// Build: mpicxx ex05_1_mismatch.cc -o ex05_1_mismatch
// Run  : timeout 15 mpirun -np 2 ./ex05_1_mismatch 1

#include <mpi.h>
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int mode = (argc > 1) ? atoi(argv[1]) : 1;
    int number = 0;

    if (rank == 0) {
        number = 42;
        int dest = (mode == 3) ? size : 1;   // mode 3: rank "size" is invalid
        int tag  = 0;
        std::cout << "rank 0: sending " << number << " to rank " << dest
                  << " with tag " << tag << std::endl;
        MPI_Send(&number, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        std::cout << "rank 0: MPI_Send returned" << std::endl;
    } else if (rank == 1) {
        int src = (mode == 1) ? 1 : 0;       // mode 1: wrong source (itself)
        int tag = (mode == 2) ? 99 : 0;      // mode 2: wrong tag
        std::cout << "rank 1: waiting for a message from rank " << src
                  << " with tag " << tag << std::endl;
        MPI_Recv(&number, 1, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "rank 1: received " << number << std::endl;
    }

    MPI_Finalize();
    return 0;
}

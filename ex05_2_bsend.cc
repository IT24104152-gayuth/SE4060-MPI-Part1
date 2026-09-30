// ex05_2_bsend.cc - Exercise 5 part 2
// message2.cc rewritten with Buffered Send (MPI_Bsend).
//
// The original message2.cc reuses the single variable "number" for all three
// sends. Here each send gets its own variable (numbers[0..2]), so no send
// buffer is ever overwritten while a message is still outstanding, which is
// what the exercise asks for.
//
// A buffered send needs user memory attached first with MPI_Buffer_attach,
// sized for every message that can be in flight at once:
//     3 messages * (sizeof(int) + MPI_BSEND_OVERHEAD)
// Too small a buffer gives MPI_ERR_BUFFER at run time.
//
// Build: mpicxx ex05_2_bsend.cc -o ex05_2_bsend
// Run  : mpirun -np 2 ./ex05_2_bsend

#include <mpi.h>
#include <iostream>
#include <cstdlib>

const int MSGS = 3;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // One distinct send variable per message.
        int numbers[MSGS] = {0, 10, 20};

        int buffer_size = MSGS * (sizeof(int) + MPI_BSEND_OVERHEAD);
        char* buffer = new char[buffer_size];
        MPI_Buffer_attach(buffer, buffer_size);

        for (int i = 0; i < MSGS; i++) {
            MPI_Bsend(&numbers[i], 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            std::cout << "Process 0 sent " << numbers[i] << " (buffered)"
                      << std::endl;
        }

        // Detach blocks until every buffered message has actually left,
        // so it must come before the buffer is released.
        MPI_Buffer_detach(&buffer, &buffer_size);
        delete[] buffer;
    } else if (rank == 1) {
        int number;
        for (int i = 0; i < MSGS; i++) {
            MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Process 1 received " << number << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}

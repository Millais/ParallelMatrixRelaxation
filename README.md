## Parallel Matrix Relaxation with C

The matrix relaxation technique continually replaces each non-edge cell in a matrix with the average of its 4 neighbours, until a pre-defined precision is reached. This technique was implemented in both shared and distributed memory architectures (as part of [CM30225](http://people.bath.ac.uk/masrjb/CourseNotes/cm30225.html): *Parallel Computing*).

Each C program was executed on Balena – the HPC cluster at the University of Bath – using SLURM with jobs submitted in varying configurations to support the scalability and efficiency investigations inside the technical reports.

Both assignments were awarded strong firsts. 

______

The full technical reports are available in this repo. A summary of each assignment is included below. 

### Shared Memory Architecture

* Implemented using [POSIX](https://en.wikipedia.org/wiki/POSIX) Threads, and tested using up to 32 threads on Balena. Testing largely took place with 16 threads, as each node on Balena had 16 cores supporting a single thread each. See 4.2 in the Technical Report for a further description on idling threads and the communication overhead.
* Scalability of initial starting cells optimised through [KCachegrind](http://kcachegrind.sourceforge.net/html/Home.html) profiling.
* Thread pool design pattern to avoid expensive creation/destruction pthread lifecycle operations.
* Superstep model using worker threads synchronising at barriers.
* Minimal global state maintained. Scalability limitation of cacheline bouncing discussed in 2.2 of the Technical Report.
* Extensive exploration into processing power scalability and problem size scalability. Coherency with Amdahl's Law and Gustafson's Law investigated in 4, alongside additional discussion on the observation of superlinear speedup.

### Distributed Memory Architecture

* Implemented using [Open MPI](https://www.open-mpi.org/) to yield parallelism via the Single Program, Multiple Data technique.
* Use of the Scatter-Gather approach to distribute work aross multiple processes.
* Resultantly, uses ~2.5% of the network bandwidth which an equivalent broadcasting approach uses.
* Scoping precision checking optimisation (e.g with *MPI_Reduce/MPI_Sum*)
* Investigation into speedup and efficiency across 64 processes on 4 16-core nodes.
* Asymmetric limitation of Balena job scheduling explored – proximity of nodes and effect on communication costs.

____

### Running The Program

See the technical report for further information on command line arguments. 

###### Compiling 

> gcc -std=gnu99 -Wall -pthread -o shared_parallel shared_parallel.c

> mpicc -Wall -std=gnu99 mpi distributed_parallel.c -o mpi distributed_parallel

###### Execution

> ./shared_parallel -S 10 -P 0.1 -T 2 -V

> mpirun -np 3 mpi distributed_parallel -S 10 -P 0.01 -V



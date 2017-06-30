#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <getopt.h>

/* environment variables */
static const char *opt_string = "S:P:T:V";
struct {
    int array_size;
    double precision;
    int verbosity;
} args;

void setup_environment(int, char**, const char*, int);
void init_array(double*, int);
int on_edge(int, int);
void relax_inner(double*, double*, int, int);
double calc_average(double, double, double, double);
void print_arrays(double*, double*, int);
int check_precision(double*, double*, int, double);
void swap_pointers(double**, double**);

int main(int argc, char **argv){

	/* get rank of current process and the number of total processes */
    int ierr, global_processes, rank;
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &global_processes);
	if (args.verbosity){ printf("Starting distributed program\n");}

	/* set environment defaults */
	args.array_size = 10;
	args.precision = 0.01;
	args.verbosity = 0;
	setup_environment(argc, argv, opt_string, rank);

	/* allocate memory for both matrices */
	double *a = malloc(args.array_size*args.array_size*sizeof(double));
	double *b = malloc(args.array_size*args.array_size*sizeof(double));
	if (a == NULL || b == NULL){ 
		printf("Couldn't allocate memory");
	}

	/* fill edge of arrays with 1s and inner cells with 0s  */
	init_array(a, args.array_size);
	init_array(b, args.array_size);

	/* print initialised arrays */
	if (rank == 0 && args.verbosity){
		printf("Initialised Arrays\n");
		print_arrays(a, b, args.array_size);
	}

	/* allocate arrays for communication between processes */
	int *displacement = malloc((global_processes)*sizeof(int));
	int *send_count = malloc((global_processes)*sizeof(int));
	int *gather_displacement = malloc((global_processes)*sizeof(int));
	int *gather_receive_count = malloc((global_processes)*sizeof(int));

	/* each process receives a standard number of rows*/
	int base_rows = ((args.array_size-2) / global_processes);

	for (int i = 0; i < global_processes; i++){

		/* starting position of each chunk to send */
		displacement[i] = i * base_rows * args.array_size;
		/* starting position of each chunk to gather */
		gather_displacement[i] = args.array_size +
									(i * base_rows * args.array_size);

		/* the work on the last process is specific (and potentially unequal) */
		if (i == global_processes-1){
			/* send the last process a specific amount of work */
			send_count[i] = (base_rows + 
							((args.array_size-2) % global_processes) + 2) 
							* args.array_size;
			/* gather a specific amount of work on the last process */
			gather_receive_count[i] = (base_rows + 
									((args.array_size-2) % global_processes)) *
									args.array_size;
		}else{
			/* send an equal amount of work to each (non-last) process */
			send_count[i] = (base_rows + 2) * args.array_size;
			/* gather an equal amount of work from each (non-last) process */
			gather_receive_count[i] = base_rows * args.array_size;
		}
	}

	/* allocate send & receive buffers */
	/* use rank of current process to calculate the size of the buffers */
	int recv_size, send_size;

	if (rank == (global_processes-1)){

		/* the last process has unequal buffer sizes */
		recv_size = (base_rows + 
			((args.array_size-2) % global_processes) + 2) * args.array_size;

		send_size = (base_rows + 
			((args.array_size-2) % global_processes)) * args.array_size;
			
	}else{
		/* all other processes have equal buffer sizes */
		recv_size = (base_rows + 2) * args.array_size;
		send_size = base_rows * args.array_size;
	}

	/* allocate buffer on each process using the pre-calculated size */
	double *recvbuf = malloc(recv_size * sizeof(double));
	double *sendbuf = malloc(send_size * sizeof(double));    

	int outside_threshold = 1;
	int iterations = 0;

	while (1){
		
		/* scatter from root process into the buffers of all other processes */
	    MPI_Scatterv(a, send_count, displacement, MPI_DOUBLE, recvbuf, 
	    				recv_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		/* each process relaxes their own section held in their receive buffer 
		   into the send buffer */
		relax_inner(recvbuf, sendbuf, args.array_size, recv_size);

		/* root node gathers relaxed data from all processes and reconstructs
		   into matrix b */
		MPI_Gatherv(sendbuf, send_size, MPI_DOUBLE, b, gather_receive_count,
					gather_displacement, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		/* check the precision between matrices on the root node only
		   – see report for justification */
		if (rank == 0){
			outside_threshold = check_precision(a, b, args.array_size,
														args.precision);
		}

		/* broadcast threshold from root to other processes so they know whether
		   to continue or exit */
		MPI_Bcast(&outside_threshold, 1, MPI_INT, 0, MPI_COMM_WORLD);

		/* swap matrices for next iteration */
		swap_pointers(&a, &b);
		iterations++;

		/* stop relaxing if precision has been reached */
		if (!outside_threshold){
			break;
		}
	}

	/* print finishing statement */
	if (rank == 0){
		if (args.verbosity){
			printf("Relaxed Arrays\n");
			print_arrays(a, b, args.array_size);
		}
		printf("Finished in %d iterations\n", iterations);
	}

	/* safely exit from all processes */
	ierr = MPI_Finalize();

	/* deallocate memory */
	free(a);
	free(b);
	free(recvbuf);
	free(sendbuf);
	free(displacement);
	free(send_count);
	free(gather_displacement);
	free(gather_receive_count);

	return ierr ? 1 : 0;
   
}
/* relax the targeted section of each processes buffer*/
void relax_inner(double* recvbuf, double* sendbuf, int dimension, 
				 int recv_size){

	/* start relaxation from the second row */
	int start = dimension;
	/* stop relaxing a row from the end */
	int end = recv_size - dimension - 1;
	/* keep track of where to store relaxed results in send buffer*/
	int count = 0;

	/* iterate through the given section */
	for (int i = start; i <= end; i++){
		if ((i % dimension != 0) && ((i+1) % dimension != 0)){
			/* relax non-side-edge values and put them into the send buffer */
			sendbuf[count] = calc_average(	recvbuf[i-1], recvbuf[i+1],
											recvbuf[i+dimension],
											recvbuf[i-dimension]);
		}else{
			/* don't relax side-edge values */
			sendbuf[count] = recvbuf[i];
		}
		count++;
	}
}

/* returns an average of the provided cells */
double calc_average(double a, double b, double c, double d){
	return (a + b + c + d)/4;
}

/* swap pointers of the target matrices */
void swap_pointers(double **first, double **second){
	double *temp = *first;
	*first = *second;
	*second = temp;
}

/* use getopt to assign command line arguments to environment variables */
void setup_environment(int argc, char **argv, const char* opt_string, int rank){
	int opt = getopt( argc, argv, opt_string );

	/* optarg is made available through getopt() */
    while( opt != -1 ) {
        switch( opt ) {

    	    case 'P':
            	args.precision = atof(optarg);
            break;

            case 'S':
                args.array_size = atoi(optarg);
                break;

            case 'V':
                args.verbosity++;
                break;
                
            default:
                break;
        }        
        opt = getopt( argc, argv, opt_string );
    }

	if (rank == 0 && args.verbosity){
    	printf("\nArray size: %d. ", args.array_size);
		printf("Precision: %f\n", args.precision);
    }
}

/* fill edge cells with 1s, inner cells with 0s */
void init_array(double *m, int dimension){
	for (int i = 0; i < (dimension*dimension); i++){
		if (on_edge(i, dimension)){
			m[i] = 1.0;
		}else{
			m[i] = 0.0;
		}
	}
}

/* determines whether i is on the edge of a matrix with the given dimension */
int on_edge(int i, int dimension){
	if 	(i < dimension || i % dimension == 0 ||
		(i+1) % dimension == 0 ||
		((i >= dimension * (dimension-1)) && (i < dimension * dimension))){
			return 1;
		}
	return 0;
}

/* prints both matrices */
void print_arrays(double *a, double *b, int dimension){
	
	printf("\n-- Array A --\n");

	for (int i = 0; i < (dimension * dimension); i++){
		if (i % dimension == 0){
			printf("\n");
		}
		printf("%f ",a[i]);
	}

	printf("\n-- Array B --\n");

	for (int i = 0; i < (dimension * dimension); i++){
		if (i % dimension == 0){
			printf("\n");
		}
		printf("%f ",b[i]);
	}

	printf("\n-------------\n");
}

/* check precision between the two given matrices in associated cells */
int check_precision(double *a, double *b, int size, double precision){

	int i;
	for (i = 0; i < (size * size); i++){
		if (fabs(a[i]-b[i]) > precision){
			return 1;
		}
	}
	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <pthread.h>

/* environment variables */
struct {
    int array_size;
    double precision;
    int n_of_threads; 
    int verbosity;
} args;

/* thread data */
typedef struct {
	int thread_id;
	int n_cells;
	int start_i;
	int start_j;
	double **a_ptr;
	double **b_ptr;
} thr_data;

void init_array(double**, int);
void init_array_edge(double**, int);
void print_arrays(double**, double**);
void pretty_print_arrays(double**, double**, int);
double calc_average(double**, int, int);
int check_precision(double**, double**, int, int);
int check_aggregated_precision(char*, int);
void setup_environment(int, char**, const char*);
void determine_thread_data(thr_data*, double**, double**);
void calculate_start_position(thr_data*);
void swap_pointers(thr_data*);

static const char *opt_string = "S:P:T:V";
pthread_barrier_t   barrier;
char *precision;
int outside_threshold;


/* Relaxes the given segment of the matrix and writes to the second matrix
 * Reports back to the main thread whether precision is reached for its segment
 * Returns with null, enabling threads to be joined in the main thread
 */
void *relax_segment(void *arg) {
	thr_data *data = (thr_data *)arg;

	if(args.verbosity){
		printf("Thread %d, starting at (%d,%d) for %d cells",
	 			data->thread_id, data->start_i, data->start_j, data->n_cells);
	}

	int i, j, p;

	while(1){

		/* reset variables for new section*/
		i = data ->start_i;
		j = data ->start_j;
		p = 1;

		for (int c=0; c < data->n_cells; c++){
			/* write average to the second matrix */
			data->b_ptr[i][j] = calc_average(data->a_ptr, i, j);

			/* p is assumed to be in precision, change if it isn't */
			if(!check_precision(data->a_ptr,data->b_ptr, i, j)){
				p = 0;
			} 

			/* get next cell to work on */
			j++;
			if (j == (args.array_size -1)){
				i++;
				j = 1;
			}
		}

		/* record if all cells are in precision to global array */
		precision[data->thread_id] = p;

		if (args.verbosity){
			printf("\nThread: %d: P reached: %d Starting at [%d,%d] for %d\n",
					data-> thread_id, precision[data-> thread_id], 
					data->start_i, data->start_j, data->n_cells);
		}

		/* synchronise */
		pthread_barrier_wait (&barrier);
		pthread_barrier_wait (&barrier);

		if (outside_threshold){
			if (args.verbosity){printf("Thread %d: Continuing\n",
								data->thread_id);}
		}else{
			break; 	/* precision reached */
		}

		swap_pointers(data);

	}

	if (args.verbosity){printf("Thread %d: Terminating\n", data->thread_id);}
	pthread_exit(NULL);
}

int main(int argc, char **argv){

	//srand((unsigned)time((1));
	int iterations = 0;

	/* set environment defaults */
	args.array_size = 6;
	args.precision = 0.1;
	args.n_of_threads = 2;
	args.verbosity = 0;
	setup_environment(argc, argv, opt_string);

	double **a = malloc(args.array_size*sizeof(double*));
	double *buf = malloc(args.array_size*args.array_size*sizeof(double));
	double **b = malloc(args.array_size*sizeof(double*));
	double *newbuf = malloc(args.array_size*args.array_size*sizeof(double));
 	thr_data *t_data = malloc(args.n_of_threads*sizeof(thr_data));

	if (a == NULL || buf == NULL || b == NULL || 
		newbuf == NULL || t_data == NULL){ 
		printf("Couldn't allocate memory");
	}

	for (int i = 0; i < args.array_size; i++) {
		a[i] = buf + args.array_size*i;
		b[i] = newbuf + args.array_size*i;
	}

	init_array(a, args.array_size);
	init_array_edge(b, args.array_size);
 	determine_thread_data(t_data, a, b);
	calculate_start_position(t_data);

	outside_threshold = 1;
	int rc;
	pthread_t *threads = malloc(args.n_of_threads*sizeof(pthread_t));
	precision = malloc(args.n_of_threads*sizeof(char));
	if (threads == NULL || precision == NULL){
		printf("Couldn't allocate memory for threads");
	}

	/* synchronise threads using this barrier, +1 for controlling main thread */
	pthread_barrier_init (&barrier, NULL, args.n_of_threads + 1);

	for (int i=0; i < args.n_of_threads; i++){
		if ((rc = pthread_create(&threads[i], NULL, 
										relax_segment, &t_data[i]))){
			printf("Error creating thread");
		}
	}

	while (1){

		if (args.verbosity){
			pretty_print_arrays(a, b, iterations);
		}

		iterations++;

		pthread_barrier_wait (&barrier);

		if (check_aggregated_precision(precision, args.n_of_threads)){
			if (args.verbosity){printf("\nMAIN THREAD: Precision reached\n");}
			outside_threshold = 0;
			pthread_barrier_wait(&barrier); /* sync threads and exit */
			break;
		}

		if (args.verbosity){
			printf("\nMAIN THREAD: Imprecise, synchronising threads\n");
		}
		pthread_barrier_wait(&barrier); /* sync threads and loop */
	}

	/* block main thread until worker threads have all terminated */
	for (int i = 0; i < args.n_of_threads; i++) {
		pthread_join(threads[i], NULL);
	}

  	if (args.verbosity){
  		printf("\nMAIN THREAD: All threads terminated.\n\n");
		pretty_print_arrays(a, b, -1);
	}

	printf("\nCompleted in %d iterations\n", iterations);
	printf("Matrix: %dx%d. Threads %d. Precision: %f\n", args.array_size, 
			args.array_size, args.n_of_threads, args.precision);

	/* deallocate memory */
	free(a);
	free(b);
	free(buf);
	free(newbuf);
	free(t_data);
	free(threads);
	free(precision);
}

/* use getopt to assign command line arguments */
void setup_environment(int argc, char **argv, const char* opt_string){
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

            case 'T':
                args.n_of_threads = atoi(optarg);
                break;

            case 'V':
                args.verbosity++;
                break;
                
            default:
                break;
        }        
        opt = getopt( argc, argv, opt_string );
    }

    /*check number of threads <= inner cells */
    if (args.n_of_threads > ((args.array_size - 2) * (args.array_size - 2))){
    	printf("WARNING: Thread number exceeded the number of inner cells.\n");
    	args.n_of_threads = 1;
    }

    if (args.verbosity){
    	printf("\nArray size: %d\n", args.array_size);
		printf("Precision: %f\n", args.precision);
		printf("Verbosity: %d\n", args.verbosity); 
    }
}

/* swap matrices pointers in each thread's data */
void swap_pointers(thr_data *data){
	double **temp = data->a_ptr;
	data->a_ptr = data->b_ptr;
	data->b_ptr = temp;
}

/* fill thread data with the number of cells to work on and matrices pointers */
void determine_thread_data(thr_data *t_data, double **a, double **b){

	int inner_cells = (args.array_size-2) * (args.array_size-2);
 	int group_size[args.n_of_threads];  /* used on this stack frame only */

 	if (inner_cells % args.n_of_threads > 0){

 		if(args.verbosity){
 			printf("Unequally splitting %d cells into %d groups\n", 
 					inner_cells, args.n_of_threads);
 		}

 		/* e.g 20/3 = 6 and remainder 2 */
 		int base = inner_cells / args.n_of_threads;	/* truncates towards 0 */
 		int remainder = inner_cells % args.n_of_threads;

 		/* assign base size to all groups */
 		for (int i = 0; i <args.n_of_threads; i++){
 			group_size[i] = base;
 		}

 		/* add remainder one by one to groups */
 		int next_group = 0;
 		while (remainder > 0){
 			group_size[next_group] += 1;
 			remainder--;
 			if (next_group == (args.n_of_threads-1)){
 				next_group = 0;
 			}else{
 				next_group++;
 			}
 		}

 		/* assign thread data */
 		for (int i=0; i < args.n_of_threads; i++){
 			t_data[i].thread_id = i;
			t_data[i].n_cells = group_size[i];
			t_data[i].a_ptr = a;
			t_data[i].b_ptr = b;
 		}

 		if (args.verbosity){
			printf("\nSorted unequal groups: [");
			for (int i = 0; i <args.n_of_threads; i++){
				printf("%d", group_size[i]);
				if (i != (args.n_of_threads -1)){printf(", ");} /* formatting */
			}
			printf("]\n\n");
		}

 	}else{

 		if (args.verbosity){
 			printf("Equally splitting %d cells into %d groups\n", 
 					inner_cells, args.n_of_threads);
 		}

 		/* assign thread data */
 		int step = (inner_cells / args.n_of_threads);
 		for (int i=0; i < args.n_of_threads; i++){
 			t_data[i].thread_id = i;
 			t_data[i].n_cells = step;
 			t_data[i].a_ptr = a;
			t_data[i].b_ptr = b;
 		}
 	}
}


/* calculate the starting position for each thread */
void calculate_start_position(thr_data *t_data){
 	
 	/* start at first cell */
	int i = 0, j = 0, next = 0;

	for (int t=0; t < args.n_of_threads; t++){
	 	/* add 1 accounting for non-edge cells */
		t_data[t].start_i = i+1;
		t_data[t].start_j = j+1;

		/* find starting cell for next thread */
		next += t_data[t].n_cells;
		i = next / (args.array_size-2);
		j = next % (args.array_size-2);

	}
}

/* set matrix edges to fixed value, inner cells to random values between 0-1 */
void init_array(double **a, int size){
	for (int i = 0; i < size; i++){
		for (int j =0; j < size; j++){
			if (i == 0 || j == 0 || i == (size-1) || j == (size-1)){
				a[i][j] = (double)1; //use fixed val of 1 for now
			}else{
				a[i][j] = ((double)rand()/(double)RAND_MAX);
			}
		}
	}
}

/* set matrix edges to fixed value, inner cells are left alone */
void init_array_edge(double **b, int size){
	for (int i = 0; i < size; i++){
		b[i][0] = (double)1;
		b[0][i] = (double)1;
		b[size-1][i] = (double)1;
		b[i][size-1] = (double)1;
	}
}

/* returns average of the cells adjacent to the target cell */
double calc_average(double **target, int i, int j){
	return (target[i-1][j]+target[i+1][j]+target[i][j-1]+target[i][j+1])/4;
}

/* returns 0 if cells aren't in precision, 1 otherwise */
int check_precision(double **a, double **b, int i, int j){
	if (fabs(a[i][j]-b[i][j]) > args.precision){
		return 0;
	}
	return 1;
}

/* checks elements of the precision array corresponding to matrix segments */
/* returns 0 if a single cell isn't in precision, 1 otherwise */
int check_aggregated_precision(char *precision, int length){
	for (int i=0; i <length; i++){
		if (!precision[i]){
			return 0;
		}
	}
	return 1;
}

/* prints both matrices together */
void print_arrays(double **a, double **b){
	printf("Matrix A                                          Matrix B\n");
	for (int i = 0; i < args.array_size; i++) {
		for (int j = 0; j < args.array_size; j++) {
			printf("%f ", a[i][j]);
		}
		printf("     ");
		for (int j = 0; j < args.array_size; j++) {
			printf("%f ", b[i][j]);
		}
		printf("\n");
	}
}

/* prints both matrices with a wrapper for the iteration count
 * passing an iteration number below 0 enables the final matrices text
 */
void pretty_print_arrays(double **a, double **b, int iterations){
	if (iterations < 0){
		printf("--------------Final Matrices--------------\n");
	}else{
		printf("\n---------Matrices at iteration %d---------\n", iterations);
	}
	print_arrays(a, b);
	printf("------------------------------------------\n");
}


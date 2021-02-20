#define CUB_IGNORE_DEPRECATED_CPP_DIALECT 1 //ignore warnings caused by old compiler version
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT 1

#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <sstream>
#include <assert.h>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

//file for handling command line arguments
#include "args.hxx"

#define DELTA 0.5

struct Event
{
	double s12;
	double s13;
	double s24;
	double s34;
	double s134;
	double hmg;//helpful variable to reduce time needed for calculating euclidean distance
};

/**
Energy test wrapper, calls other functions based on command line arguments
@param argc number of command line arguments
@param argv array of command line arguments
*/
int energy_test_gpu_wrapper(int argc, char ** argv);

/**
Reads data from file and returns it as thrust::host_vector
@param file_name name of the file to read data from
@param events number of events to read
*/
thrust::host_vector<Event> load_data(const std::string & file_name, const size_t events);

/**
Calculates the T value of 2 dataset
@param cpu_dataset_1 first dataset
@param cpu_dataset_2 second dataset
@param size_1 number of events in 1st dataset
@param size_2 number of events in 2nd dataset
@param prop structure with active device's properties
@param show_distances display distances
*/
double compute_statistic(const thrust::host_vector<Event> & cpu_dataset_1, const thrust::host_vector <Event> & cpu_dataset_2, int size_1, int size_2,  cudaDeviceProp & prop, bool show_distances=true);

/**
Computes individual statistic contributions(Ti values) and returns them in array
@param cpu_dataset_1 1st dataset, Ti values are related with this set
@param cpu_dataset_2 2nd dataset
@param size_1 number of events in 1st dataset
@param size_2 number of events in 2nd dataset
@param prop structure with active device's properties 
*/
double * compute_statistic_contributions(const thrust::host_vector<Event> & cpu_dataset_1, const thrust::host_vector<Event> & cpu_dataset_2, int size_1, int size_2, cudaDeviceProp & prop);


/**
Device kernel that calculates distance between 2 datasets
@param dataset_1 1st dataset
@param dataset_2 2nd dataset
@param size_1 number of elements in 1st dataset
@param size_2 number of elements in 2nd dataset
@param sum pointer to variable storing the distance between 2 datasets
@param shared_el number of elements from 2nd dataset stored in each block's shared memory
@param same_sets if true, datasets are the same, do not compute distance between same elements
*/
__global__
void compute_distance(const Event * dataset_1, const Event * dataset_2, const int size_1, const int size_2, double * sum, const int shared_el, bool same_sets=true);

/**
Device kernel that calculates distance between dataset and single event
@param dataset dataset
@param ev single event
@param size_1 number of events in dataset
@param sum pointer to variable storing the distance
@param shared_el number of elements stored each block's shared memory
@param index_to_skip index of ev (if ev is from dataset) used to skip it's distance to itself, otherwise -1
*/ 
__global__
void compute_individual_distance(const Event * dataset, const Event * ev, const int size_1, double * sum, int shared_el, int index_to_skip=-1);

/**
Function checking for CUDA runtime errors
@param result error code returns by CUDA functions
*/
inline cudaError_t checkCudaError(cudaError_t result);


int main(int argc, char *argv[])
{
	try
	{
		return energy_test_gpu_wrapper(argc, argv);
	}
	catch(std::runtime_error & err)
	{
		std::cerr << err.what() << std::endl;
		exit (EXIT_FAILURE);
	}
	
	return 0;
}

int energy_test_gpu_wrapper(int argc, char ** argv)
{

	// Parsing command line arguments, setting up execution options
	args::ArgumentParser parser("GPU based energy test");
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::Flag calculate_ti(parser, "calculate ti", "Calculate individual contributions to test statistic", {"calculate-ti"});
 	args::Flag permutations_only(parser, "permutations only", "Only calculate permutations", {"permutations-only"});
 	args::Flag output_write(parser, "output write", "write output Tvalues", {"output-write"});
 	args::ValueFlag<size_t> n_permutations(parser, "n_permutations", "Number of permutations to run", {"n-permutations"});
 	args::ValueFlag<size_t> max_events_1(parser, "max events 1", "Maximum number of events to use from dataset 1", {"max-events-1"});
 	args::ValueFlag<size_t> max_events_2(parser, "max events 2", "Maximum number of events to use from dataset 2", {"max-events-2"});
 	args::ValueFlag<size_t> max_events(parser, "max events", "Max number of events in each dataset", {"max-events"});
 	args::ValueFlag<size_t> seed(parser, "seed", "seed for permutations", {"seed"});
 	args::ValueFlag<size_t> max_permutation_events_1(parser, "max permutation events 1", "Max number of events in dataset 1 for permutations",
                                                   {"max-permutation-events-1"});
 	args::ValueFlag<std::string> ti_output_fn_1(parser, "ti output filename 1", "Filename for individual contributions to test statistic from dataset 1", {"ti-output-fn-1"});
 	args::ValueFlag<std::string> ti_output_fn_2(parser, "ti output filename 2", "Filename for individual contributions to test statistic from dataset 2", {"ti-output-fn-2"});
 	args::ValueFlag<std::string> permutation_ti_minmax_output_fn(parser, "permutation ti min-max filename", "Output filename for the minimum and maximum Ti values from permutations",
								{"permutation-ti-minmax-output-fn"});
 	args::Positional<std::string> filename_1(parser, "dataset 1", "Filename for the first dataset");
 	args::Positional<std::string> filename_2(parser, "dataset 2", "Filename for the second dataset");
 	args::Positional<std::string> permutation_output_fn(parser, "permutation output filename", "Output filename for the permutation test statistics", {"permutation-output-fn"});

	try
	{
		parser.ParseCLI(argc, argv);
		
		//Check for neccessary options
		if (!filename_1 || !filename_2)
		{
			throw args::ParseError("Direct paths to two dataset files must be given");
		}
		if ((max_events_1 || max_events_2) && max_events)
		{
			throw args::ParseError("--max-events cannot be used with --max-events-1 or --max-events-2");
		}
		if (calculate_ti && max_permutation_events_1)
		{
			throw args::ParseError("--calculate-ti cannot be used with --max-permutation-events-1");
		}
	}
	catch (args::Help)
	{
		std::cout << parser;
		return 0;
	}
	catch (args::ParseError & err)
	{
		std::cerr << err.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

        // Displaying information about the device
        int deviceId;
        checkCudaError(cudaGetDevice(&deviceId));

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);

        printf("\nDevice name: %s\n", prop.name);
        printf("Shared memory per block: %zu\n", prop.sharedMemPerBlock);
        printf("Threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Number of SMs: %d\n", prop.multiProcessorCount);
        printf("Shared memory per SM: %zu\n", prop.sharedMemPerMultiprocessor);
        printf("Max Block per SM: %d\n", prop.maxBlocksPerMultiProcessor);

        int val;
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxThreadsPerBlock, deviceId);
        printf("Max Threads per SM: %d\n\n", val);


	// Load specified number of events from data files
	size_t data_1_limit, data_2_limit;
	data_1_limit = data_2_limit = std::numeric_limits<size_t>::max();

	if (max_events)
	{
		data_1_limit = data_2_limit = args::get(max_events);
	}	
	else
	{
		if (max_events_1)
		{
			data_1_limit = args::get(max_events_1);
		}
		if (max_events_2)
		{
			data_2_limit = args::get(max_events_2);
		}
	}

	//Read data from files
	thrust::host_vector<Event>dataset_1 = load_data(args::get(filename_1), data_1_limit);
	thrust::host_vector<Event>dataset_2 = load_data(args::get(filename_2), data_2_limit);

	int size_1 = static_cast<int>(dataset_1.size());
	int size_2 = static_cast<int>(dataset_2.size());

	std::cout << "Size of dataset 1: " << size_1 << std::endl;
	std::cout << "Size of dataset 2: " << size_2 << std::endl << std::endl;

	double t_value;

	if (!permutations_only)
	{
		// Calculate individual contrubutions to the test (Ti values)
		if (calculate_ti)
		{
			std::cout << "Calculating contributions of individual events to test statistic..." << std::endl;
			double * tis_1 = compute_statistic_contributions(dataset_1, dataset_2, size_1, size_2, prop);
			double * tis_2 = compute_statistic_contributions(dataset_2, dataset_1, size_2, size_1, prop);
			
			double total = 0;
			for (int i=0; i < size_1; i++) 
				total += tis_1[i];
			for (int i=0; i < size_2; i++) 
				total += tis_2[i];

			t_value = total;
     			std::cout << "Test statistic for nominal dataset:" << std::endl << "  T = " << t_value << std::endl << std::endl;
		
	                //write Ti values to file
        	        std::string ti_file_1;
                	std::string ti_file_2;

                	if (ti_output_fn_1)
                	{
                        	ti_file_1 = args::get(ti_output_fn_1);
                	}
                	else
               		{
				ti_file_1 = "Ti_dataset_1_" + std::to_string(size_1) + ".txt";
			}

	                if (ti_output_fn_2)
			{
                        	ti_file_2 = args::get(ti_output_fn_2);
                	}
               		else
                	{
                        	ti_file_2 = "Ti_dataset_2_" + std::to_string(size_2) + ".txt";
                	}

                	std::ofstream file1(ti_file_1);
                	if (!file1.is_open())
                	{
                        	throw std::runtime_error("Cannot open file" + ti_file_1);
                	}

                	for (int i=0; i < size_1; i++)
                	{
                        	file1 << dataset_1[i].s12 << " " << dataset_1[i].s13 << " " << dataset_1[i].s24 << " ";
                        	file1 << dataset_1[i].s34 << " " << dataset_1[i].s134 << " Ti = " << tis_1[i] << std::endl;
                	}
                	file1.close();
			std::cout << "Ti values for dataset 1 written to " << ti_file_1 << std::endl;

                	std::ofstream file2(ti_file_2);
                	if (!file2.is_open())
                	{
                        	throw std::runtime_error("Cannot open file" + ti_file_2);
                	}

	                for (int i=0; i < size_2; i++)
                	{
				file2 << dataset_2[i].s12 << " " << dataset_2[i].s13 << " " << dataset_2[i].s24 << " ";
                        	file2 << dataset_2[i].s34 << " " << dataset_2[i].s134 << " Ti = " << tis_2[i] << std::endl;
                	}
			file2.close();
			std::cout << "Ti values for dataset 2 written to " << ti_file_2 << std::endl;

			free(tis_1);
			free(tis_2);
		}
		// calculate T value for nominal datasets
		else
		{
			t_value = compute_statistic(dataset_1, dataset_2, size_1, size_2, prop);
			std::cout << "\nT  = " << t_value << std::endl;
		}
		
	}

	if (n_permutations)
	{
		// Vector for permutations of events
		thrust::host_vector<Event>permuted_events;
		permuted_events.insert(permuted_events.end(), dataset_1.begin(), dataset_1.end());
		permuted_events.insert(permuted_events.end(), dataset_2.begin(), dataset_2.end());
	
		// Number of permutations to run
		int N = static_cast<int>(args::get(n_permutations));

		// Number of events used in permutation	
		int n_events_1 = size_1;
		int n_events_2 = size_2;

		if (max_permutation_events_1)
		{
			n_events_1 = std::min(n_events_1, static_cast<int>(args::get(max_permutation_events_1)));
			n_events_2 = std::round(n_events_1 * (static_cast<double>(size_2) / static_cast<double>(size_1)));
		}

		double factor = static_cast<double>(n_events_1 + n_events_2) / static_cast<double>(size_1 + size_2);

		// Random generator for shuffling
		int random_seed = static_cast<int>(seed ? args::get(seed) : std::mt19937::default_seed);
		thrust::default_random_engine random_generator(random_seed);

		// Output to files

		//T values
		std::string T_file;
		if (permutation_output_fn)
		{
			T_file = args::get(permutation_output_fn);
		}
		else
		{
			T_file = "T_values_permutations_" + std::to_string(size_1) + "_" + std::to_string(size_2) + "_" + std::to_string(random_seed) + ".txt";
		}
		std::ofstream t_file;
		if (output_write || permutation_output_fn)
		{
			t_file.open(T_file);
		}

		//T min/max values
		std::string T_minmax_file;
		if (permutation_ti_minmax_output_fn)
		{
			T_minmax_file = args::get(permutation_ti_minmax_output_fn);
		}
		else
		{
			T_minmax_file = "T_minmax_permutations_" + std::to_string(size_1) + "_" + std::to_string(size_2) + "_" + std::to_string(random_seed) + ".txt";
		}
		std::ofstream t_minmax_file;
		if (output_write || permutation_ti_minmax_output_fn)
		{
			t_minmax_file.open(T_minmax_file);
		}

		// p values
		std::ofstream p_file;
		if (!permutations_only)
		{
			p_file.open("pvalues.txt", std::iostream::out | std::iostream::app);//append if file exists
		}

		int nsig = 0;
		int skip = permuted_events.size()+1;
		std::cout << "Calculating " << N << " permutations" << std::endl;

		for (int i=0; i < N; i++)
		{
			if (skip + n_events_1 + n_events_2 > static_cast<int>(permuted_events.size()))//avoid unneccesary shuffling
			{
				thrust::shuffle(permuted_events.begin(), permuted_events.end(), random_generator);
				skip = 0;
			}

			//get permuted datasets
			thrust::host_vector<Event>permuted_set_1(permuted_events.begin() + skip, permuted_events.begin() + n_events_1 + skip);
			skip += n_events_1;
			thrust::host_vector<Event>permuted_set_2(permuted_events.begin() + skip, permuted_events.begin() + n_events_2 + skip);
			skip += n_events_2;

			double test_statistic;

			// Calculate Ti values for permuted set
			if (calculate_ti)
			{
				double * tis_1 = compute_statistic_contributions(permuted_set_1, permuted_set_2, n_events_1, n_events_2, prop);
				double * tis_2 = compute_statistic_contributions(permuted_set_2, permuted_set_1, n_events_2, n_events_1, prop);

				double total = 0;
				for (int i=0; i < n_events_1; i++)
					total += tis_1[i];
				for (int i=0; i < n_events_2; i++)
					total += tis_2[i];
				test_statistic = total;

				// Find min and max Ti values
				double ti_min = tis_1[0];
				double ti_max = tis_1[0];

				for (int i=1; i < n_events_1; i++)
				{
					if (tis_1[i] > ti_max)
						ti_max = tis_1[i];
					if (tis_1[i] < ti_min)
						ti_min = tis_1[i];
				}

				for (int i=0; i < n_events_2; i++)
				{
					if (tis_2[i] > ti_max)
						ti_max = tis_2[i];
					if (tis_2[i] < ti_min)
						ti_min = tis_2[i];
				}

				if (output_write || permutation_ti_minmax_output_fn)
				{
					t_minmax_file << ti_min << " " << ti_max << std::endl;				
				}

				free(tis_1);
				free(tis_2);	
			}
			else
			{
				test_statistic = compute_statistic(permuted_set_1, permuted_set_2, n_events_1, n_events_2, prop, false);
			}

			if (output_write || permutation_output_fn)
			{
				t_file << test_statistic << std::endl;
			}
			
			if (!permutations_only)
			{
				if (factor * test_statistic > t_value)
					nsig++; //used to calculate p-value
			}
		}
		if (!permutations_only)
		{
			double p_value = static_cast<double>(nsig)/static_cast<double>(N);
			p_file << DELTA << " " << p_value << std::endl;
			std::cout << "p-value = " << p_value << std::endl;
		}

		t_file.close();
		t_minmax_file.close();		
		p_file.close();
		
	}
	
	return 0;	
}

thrust::host_vector<Event> load_data(const std::string & file_name, const size_t max_size)
{
	std::ifstream file(file_name);
	if (!file.is_open())
	{
		throw std::runtime_error("Cannot open file" + file_name);
	}

	thrust::host_vector<Event> events;
	events.reserve(std::min(max_size, static_cast<size_t>(500000)));	

	std::string line;
	while (std::getline(file, line) && events.size() < max_size)
	{
		Event e;
		std::istringstream is (line);

		//extracting each line for double values
		is >> e.s12 >> e.s13 >> e.s24 >> e.s34 >> e.s134;
		if (is.fail())
		{
			throw std::runtime_error("Error reading line in " + file_name);
		}
		e.hmg = 0.5 * (e.s12*e.s12 + e.s13*e.s13 + e.s24*e.s24 + e.s34*e.s34 + e.s134*e.s134);
		events.push_back(e);
	}

	return events;
}

double compute_statistic(const thrust::host_vector<Event> &  cpu_dataset_1, const thrust::host_vector <Event> & cpu_dataset_2, int size_1, int size_2, cudaDeviceProp & prop, bool show_distances)
{
	//allocating device memory
	thrust::device_vector<Event> gpu_dataset_1 = cpu_dataset_1;
	thrust::device_vector<Event> gpu_dataset_2 = cpu_dataset_2;

	double dist_11, dist_22, dist_12;
	double *d_dist_11, *d_dist_22, *d_dist_12;

	checkCudaError(cudaMalloc(&d_dist_11, sizeof(double)));
        checkCudaError(cudaMalloc(&d_dist_22, sizeof(double)));
        checkCudaError(cudaMalloc(&d_dist_12, sizeof(double)));

	checkCudaError(cudaMemset(d_dist_11, 0.0, sizeof(double)));
        checkCudaError(cudaMemset(d_dist_22, 0.0, sizeof(double)));
        checkCudaError(cudaMemset(d_dist_12, 0.0, sizeof(double)));
	
	//set up execution configuration
	int blocks = 1;
	int threads = 1024;

	int shared_el;
	int max_shared_per_block = prop.sharedMemPerBlock;

	//distance 1_1
	shared_el = size_1 / blocks;
	if (size_1 % blocks)
		shared_el++;

	while (shared_el * static_cast<int>(sizeof(Event)) > max_shared_per_block)
	{
		blocks *= 2;
		shared_el = size_1 / blocks;
		if (size_1 % blocks)
			shared_el++;
	}
	compute_distance<<<blocks, threads, shared_el * sizeof(Event)>>>(thrust::raw_pointer_cast(gpu_dataset_1.data()), thrust::raw_pointer_cast(gpu_dataset_1.data()), size_1, size_1, d_dist_11, shared_el, true);

	//distance 2_2
	blocks = 1;
	threads = 1024;

        shared_el = size_2 / blocks;
        if (size_2 % blocks)
                shared_el++;

        while (shared_el * static_cast<int>(sizeof(Event)) > max_shared_per_block)
        {
                blocks *= 2;
                shared_el = size_2 / blocks;
                if (size_2 % blocks)
                        shared_el++;
        }

        compute_distance<<<blocks, threads, shared_el * sizeof(Event)>>>(thrust::raw_pointer_cast(gpu_dataset_2.data()), thrust::raw_pointer_cast(gpu_dataset_2.data()), size_2, size_2, d_dist_22, shared_el, true);

	//distance 1_2
        blocks = 1;
        threads = 1024;

        shared_el = size_2 / blocks;
        if (size_2 % blocks)
                shared_el++;

        while (shared_el * static_cast<int>(sizeof(Event)) > max_shared_per_block)
        {
                blocks *= 2;
                shared_el = size_2 / blocks;
                if (size_2 % blocks)
                        shared_el++;
        }	

        compute_distance<<<blocks, threads, shared_el * sizeof(Event)>>>(thrust::raw_pointer_cast(gpu_dataset_1.data()), thrust::raw_pointer_cast(gpu_dataset_2.data()), size_1, size_2, d_dist_12, shared_el, false);

	checkCudaError(cudaDeviceSynchronize());

	checkCudaError(cudaMemcpy(&dist_11, d_dist_11, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(&dist_22, d_dist_22, sizeof(double), cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy(&dist_12, d_dist_12, sizeof(double), cudaMemcpyDeviceToHost));


	dist_11 /= (2.0 * size_1 * (size_1 - 1));
	dist_22 /= (2.0 * size_2 * (size_2 - 1)); 
	dist_12 /= (1.0 * size_1 * size_2);

	if (show_distances)
	{
		std::cout << "dist_11 = " << dist_11 << std::endl;
		std::cout << "dist_22 = " << dist_22 << std::endl;
		std::cout << "dist_12 = " << dist_12 << std::endl;
	}

	checkCudaError(cudaFree(d_dist_11));
        checkCudaError(cudaFree(d_dist_22));
        checkCudaError(cudaFree(d_dist_12));
	
	return dist_11 + dist_22 - dist_12;	 
}

double * compute_statistic_contributions(const thrust::host_vector<Event> & cpu_dataset_1, const thrust::host_vector<Event> &  cpu_dataset_2, int size_1, int size_2, cudaDeviceProp & prop)
{
	double * statistic_contributions = (double *)malloc(size_1 * sizeof(double));
	if (statistic_contributions == NULL)
	{
		throw std::runtime_error("Cannot allocate host memory");
	}
	//Allocate device memory
	thrust::device_vector<Event> gpu_dataset_1 = cpu_dataset_1;
	thrust::device_vector<Event> gpu_dataset_2 = cpu_dataset_2;

        double * d_dist_e1, * d_dist_e2;
        checkCudaError(cudaMalloc(&d_dist_e1, sizeof(double)));
	checkCudaError(cudaMalloc(&d_dist_e2, sizeof(double)));

	double dist_e1, dist_e2;

	int blocks_1=1, blocks_2=1;
	int threads_1 = 1024;
	int threads_2 = 1024;

	int shared_el_1 = size_1/blocks_1;
	int shared_el_2 = size_2/blocks_2;

	if (size_1 % blocks_1) shared_el_1++;
	if (size_2 % blocks_2) shared_el_2++; 

	while ((shared_el_1+1) * sizeof(Event) > prop.sharedMemPerBlock)
	{
		blocks_1 *= 2;
		shared_el_1 = size_1/blocks_1;

        if (size_1 % blocks_1) shared_el_1++;
	}

	while ((shared_el_2+1) * sizeof(Event) > prop.sharedMemPerBlock)
	{
		blocks_2 *= 2;
		shared_el_2 = size_2/blocks_2;
	
        	if (size_2 % blocks_2) shared_el_2++;
	}

	Event * set_1_ptr = thrust::raw_pointer_cast(gpu_dataset_1.data());
	Event * set_2_ptr = thrust::raw_pointer_cast(gpu_dataset_2.data());

	for (int i=0; i < size_1; i++)
	{
		checkCudaError(cudaMemset(d_dist_e1, 0.0, sizeof(double)));
		checkCudaError(cudaMemset(d_dist_e2, 0.0, sizeof(double)));

		compute_individual_distance<<<blocks_1, threads_1, (shared_el_1 + 1) * sizeof(Event)>>>(set_1_ptr, &set_1_ptr[i], size_1, d_dist_e1, shared_el_1+1, i);
		compute_individual_distance<<<blocks_2, threads_2, (shared_el_2 + 1) * sizeof(Event)>>>(set_2_ptr, &set_1_ptr[i], size_2, d_dist_e2, shared_el_2+1);
	
		checkCudaError(cudaDeviceSynchronize());

		checkCudaError(cudaMemcpy(&dist_e1, d_dist_e1, sizeof(double), cudaMemcpyDeviceToHost));
		checkCudaError(cudaMemcpy(&dist_e2, d_dist_e2, sizeof(double), cudaMemcpyDeviceToHost));

		dist_e1 /= 2.0 * size_1 * (size_1 - 1);
		dist_e2 /= 2.0 * size_1 * size_2;

		statistic_contributions[i] = dist_e1 - dist_e2;
	}

	checkCudaError(cudaFree(d_dist_e1));
	checkCudaError(cudaFree(d_dist_e2));

	return statistic_contributions;	
}

__global__
void compute_distance(const Event * dataset_1, const Event * dataset_2, const int size_1, const int size_2, double * sum, int shared_el, bool same_sets)
{
	extern __shared__ Event event2[];

	int thread = threadIdx.x;
	int block_size = blockDim.x;
	int start = blockIdx.x * shared_el;

	while (thread+start < size_2 && thread < shared_el)
	{
		event2[thread] = dataset_2[thread+start];
		thread += block_size;
	}
	__syncthreads();

	double total = 0;
	for (int i=threadIdx.x; i < size_1; i += block_size)
	{
		const Event event1 = dataset_1[i];
		for (int j=0; j < shared_el && j+start < size_2; j++)
		{
		        if (same_sets && i == j+start)
				continue;
			//calculate half of euclidean distance squared		
			double dist = event1.hmg + event2[j].hmg - 
			(event1.s12*event2[j].s12 + event1.s13*event2[j].s13 + event1.s24*event2[j].s24 + event1.s34*event2[j].s34 + event1.s134*event2[j].s134);

			total += exp(-dist/(DELTA*DELTA));
			
		}
	}
	atomicAdd(sum, total);
}

__global__
void compute_individual_distance(const Event * dataset, const Event * ev, const int size_1, double * sum, int shared_el, int index_to_skip)
{

	extern __shared__ Event event[];
	
	int thread = threadIdx.x;
	int block_size = blockDim.x;
	int start = blockIdx.x * (shared_el-1);

	if (thread == 0)
	{
		event[0] = *ev;
	}	

        while (thread+start < size_1 && thread+1 < shared_el)
        {
                event[thread+1] = dataset[thread+start];
                thread += block_size;
        }
        __syncthreads();

	double total=0;
	for (int i=threadIdx.x+1; i < shared_el && i+start-1 < size_1; i += block_size)
	{
		if (i+start-1 == index_to_skip)
			continue;
	
		double dist = event[0].hmg + event[i].hmg - 
		(event[0].s12*event[i].s12 + event[0].s13*event[i].s13 + event[0].s24*event[i].s24 + event[0].s34*event[i].s34 + event[0].s134*event[i].s134);

		total += exp(-dist/(DELTA*DELTA));
	}
	atomicAdd(sum, total);
	
}

inline cudaError_t checkCudaError(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}





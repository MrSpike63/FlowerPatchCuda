// nvcc -o flower flower.cu -O3 -m=64 -Xptxas -O3

#include <iostream>

#define LCG_XOR 25214903917L
#define LCG_MULTIPLIER 25214903917L
#define LCG_ADDEND 11L
#define LCG_MASK (1ULL << 48) - 1


#define TOTAL_WORK (1ULL << 38)             // How many seeds to check (per task). (Each 2^48 seedspace has 6 offsets, so each offset
                                            // has a seedspace of (2^48 / 6)
#define THREAD_WORK (1ULL << 22)            // Seeds that each thread has to check
#define THREADS_PER_BLOCK 128               // Threads per kernel block
#define BLOCKS_PER_GROUP 64                 // How many blocks to do in each group (checkpoint occurs after each block group)

#define FLOWER_COUNT 18                     // Total flowers that are being matched
#define RETURN_THRESHOLD 2                  // Threshold for missing_count. missing_count < RETURN_THRESHOLD

#define DEBUG true


// Calculated with lcg_skip_calculator.cpp (calculating 2^35 lcg skips is slow)
#define LCG_SKIP_THREAD_WORK_MULTIPLIER 43720451817473ULL
#define LCG_SKIP_THREAD_WORK_ADDEND 186612394754048ULL

#define LCG_SKIP_GROUP_MULTIPLIER 121770912776193ULL
#define LCG_SKIP_GROUP_ADDEND 38139309588480ULL

#define LCG_SKIP_2_16_MULTIPLIER 184682916610049ULL
#define LCG_SKIP_2_16_ADDEND 258238410457088ULL

#define LCG_SKIP_2_32_MULTIPLIER 237099374608385ULL
#define LCG_SKIP_2_32_ADDEND 229492987527168ULL


// #define BOINC 0

#ifdef BOINC
    #include "boinc_api.h"
    #if defined _WIN32 || defined _WIN64
        #include "boinc_win.h"
    #endif
#endif

#ifndef BOINC
    #define boinc_begin_critical_section()
    #define boinc_end_critical_section()
    #define boinc_finish(status)
    #define boinc_fraction_done(fraction)
    #define boinc_checkpoint_completed()
    #define boinc_fopen(arg1, arg2) fopen(arg1, arg2)
    #define boinc_delete_file(arg1) remove(arg1)
#endif


__constant__ bool known[15][7][15];

int center[3] = {99, 68, -100};
int coords[18][3] = {
    {97, 68, -100},
    {99, 68, -102},
    {100, 68, -100}, 
    {100, 68, -99},
    {98, 68, -98},
    {100, 69, -103},
    {95, 68, -102},
    {99, 68, -98},
    {97, 69, -104},
    {96, 67, -98},
    {94, 68, -100},
    {99, 68, -97},
    {102, 68, -102},
    {100, 68, -96},
    {93, 67, -99},
    {102, 67, -96},
    {95, 70, -107},
    {105, 68, -101}
};


// Polymetrics test data
/*
int center[3] = {-715, 79, -130};
int coords[18][3] = {
    {-716,     79,   -126},
    {-713,     79,   -132},
    {-713,     79,   -129},
    {-720,     79,   -128},
    {-718,     79,   -132},
    {-712,     79,   -131},
    {-715,     79,   -128},
    {-712,     78,   -134},
    {-709,     78,   -128},
    {-717,     79,   -132},
    {-709,     78,   -131},
    {-716,     78,   -128},
    {-718,     79,   -129},
    {-713,     79,   -128},
    {-717,     79,   -130},
    {-719,     79,   -130},
    {-714,     79,   -131},
    {-715,     79,   -127},
};
*/




__device__ __forceinline__ unsigned long long next(unsigned long long *seed, short bits) {
    *seed = (*seed * LCG_MULTIPLIER + LCG_ADDEND) & LCG_MASK;
    return *seed >> (48 - bits);
}

__device__ __forceinline__ unsigned int next_int(unsigned long long *seed, short bound) {
    
    if ((bound & -bound) == bound) {
        *seed = (*seed * LCG_MULTIPLIER + LCG_ADDEND) & LCG_MASK;
        return (int)((bound * (*seed >> 17)) >> 31);
    }
    
    int bits, value;
    do {
        *seed = (*seed * LCG_MULTIPLIER + LCG_ADDEND) & LCG_MASK;
        bits =  *seed >> 17;
        value = bits % bound;
    } while (bits - value + (bound - 1) < 0);

    return value;
}


__global__ void gpu_bruteforce(unsigned long long start_seed, unsigned long long *counter, unsigned long long *seed_list, int8_t *missing_count_list, unsigned long long *lcg_skip_list_multiplier, unsigned long long *lcg_skip_list_addend) {
    // Give each thread a unique identifier (to calculate dfz offsets)
    int thread_number = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long seed = (start_seed * lcg_skip_list_multiplier[thread_number] + lcg_skip_list_addend[thread_number]) & LCG_MASK;
    // seed = seed ^ LCG_XOR;

    int8_t missing_count = FLOWER_COUNT;

    int8_t fcount[15][7][15];
    __shared__ int8_t i_buffer[64][THREADS_PER_BLOCK];
    __shared__ int8_t j_buffer[64][THREADS_PER_BLOCK];
    __shared__ int8_t k_buffer[64][THREADS_PER_BLOCK];
    
    // Initialize circular buffer
    for (int8_t i = 0; i < 64; i++) {

        // Probably not needed, but ust to be safe.
        int8_t ptr = (int8_t) (i & 63); // int ptr = i;

        int8_t i_1 = (7 + next_int(&seed, 8)) - next_int(&seed, 8);
        int8_t j_1 = (3 + next_int(&seed, 4)) - next_int(&seed, 4);
        int8_t k_1 = (7 + next_int(&seed, 8)) - next_int(&seed, 8);
        i_buffer[ptr][threadIdx.x] = i_1;
        j_buffer[ptr][threadIdx.x] = j_1;
        k_buffer[ptr][threadIdx.x] = k_1;

        if (known[i_1][j_1][k_1] && fcount[i_1][j_1][k_1] == 0) {
            missing_count--;
        }


        fcount[i_1][j_1][k_1]++;
    }

    // Do work assigned to thread
    for (long time = 0; time < THREAD_WORK; time++) {

        // Check if current flowers are a perfect match
        if (missing_count < RETURN_THRESHOLD) {
            // Find how many dfz skips were made at kernel launch
            unsigned long long dfz = (time * 6) + (thread_number * THREAD_WORK * 6);

            // Safely store in array
            atomicAdd(counter, 1);
            seed_list[*counter - 1] = dfz; // Let cpu figure the rest out lmao
            missing_count_list[*counter - 1] = missing_count;
        }

        int8_t ptr = (int8_t) (time & 63);

        // Remove previous flower
        int8_t i_0 = i_buffer[ptr][threadIdx.x];
        int8_t j_0 = j_buffer[ptr][threadIdx.x];
        int8_t k_0 = k_buffer[ptr][threadIdx.x];

        fcount[i_0][j_0][k_0]--;

        if (known[i_0][j_0][k_0] && fcount[i_0][j_0][k_0] == 0) {
            missing_count++;
        }
        
        // Add new flower
        int8_t i_1 = (7 + next_int(&seed, 8)) - next_int(&seed, 8);
        int8_t j_1 = (3 + next_int(&seed, 4)) - next_int(&seed, 4);
        int8_t k_1 = (7 + next_int(&seed, 8)) - next_int(&seed, 8); 

        i_buffer[ptr][threadIdx.x] = i_1;
        j_buffer[ptr][threadIdx.x] = j_1;
        k_buffer[ptr][threadIdx.x] = k_1;

        if (known[i_1][j_1][k_1] && fcount[i_1][j_1][k_1] == 0) {
            missing_count--;
        }

        fcount[i_1][j_1][k_1]++;
    }
}



struct checkpoint_vars {
    int offset;
    unsigned long long dfz_tracker;
    unsigned long long seed;
    time_t elapsed_chkpoint;
    unsigned long long total_counter;
};

void run_kernel(unsigned long long dfz_initial_start) {

    // Counter of seeds found per block
    unsigned long long *counter = 0;
    cudaMallocManaged(&counter, sizeof(unsigned long long));

    // List of seeds
    unsigned long long *seed_list;
    cudaMallocManaged(&seed_list, 100 * sizeof(unsigned long long));

    // The missing_list for each seed found
    int8_t *missing_count_list;
    cudaMallocManaged(&missing_count_list, 100 * sizeof(int8_t));

    // Total counter (total seeds found)
    unsigned long long total_counter = 0;




    // Calculate relative coordinates
    bool known2[15][7][15] = {false};
    for (short i = 0; i < FLOWER_COUNT; i++) {
        known2[coords[i][0] - center[0] + 7][coords[i][1] - center[1] + 3][coords[i][2] - center[2] + 7] = true;
    }

    // Pre calculate lcg skips
    unsigned long long *lcg_skip_list_multiplier;
    unsigned long long *lcg_skip_list_addend;
    cudaMallocManaged(&lcg_skip_list_multiplier, THREADS_PER_BLOCK * BLOCKS_PER_GROUP * sizeof(unsigned long long));
    cudaMallocManaged(&lcg_skip_list_addend, THREADS_PER_BLOCK * BLOCKS_PER_GROUP * sizeof(unsigned long long));
    
    // Calculate skip lists for the first kernel launch
    unsigned long long multiplier = 1;
    unsigned long long addend = 0;

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS_PER_GROUP; i++) {
        lcg_skip_list_multiplier[i] = multiplier;
        lcg_skip_list_addend[i] = addend;

        addend = (addend * LCG_SKIP_THREAD_WORK_MULTIPLIER + LCG_SKIP_THREAD_WORK_ADDEND) & LCG_MASK;
        multiplier = (multiplier * LCG_SKIP_THREAD_WORK_MULTIPLIER) & LCG_MASK;
    }


    // Copy to constant memory
    cudaMemcpyToSymbol(known, &known2, sizeof(known2));


    // Total threads needed: (total seeds to check / the seeds per thread)
    int threads_needed = TOTAL_WORK / THREAD_WORK;

    // Total blocks needed: (total threads needed / threads per block) (round up to be safe)
    int total_blocks = (threads_needed + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


    // Block group to start at
    int block_group_start = 0;
    time_t elapsed_chkpoint = 0;

    // Variables to track progress
    unsigned long long seed = 0;
    unsigned long long dfz_tracker = dfz_initial_start;

    FILE *fp;

    boinc_begin_critical_section();

    // Load checkpoint (if any)
    fp = boinc_fopen("flower_checkpoint.txt", "r");
    if (fp) {
        struct checkpoint_vars data_store;
        fread(&data_store, sizeof(data_store), 1, fp);

        block_group_start = data_store.offset;
        dfz_tracker = data_store.dfz_tracker;
        seed = data_store.seed;
        elapsed_chkpoint = data_store.elapsed_chkpoint;
        total_counter = data_store.total_counter;

        // Linux :c
        // fprintf(stderr, "Checkpoint loaded. Offset: %d, task time: %lu s.\n", block_group_start, elapsed_chkpoint);
        fprintf(stderr, "Checkpoint loaded. Offset: %d, task time: %llu s.\n", block_group_start, elapsed_chkpoint);
        fclose(fp);
    } else {
        //
        // Calculate the seed from the starting dfz given as an argument
        //

        // Large skips (2^32)
        int large_skips = floor((double)dfz_initial_start / (1ULL << 32));
        for (int i = 0; i < large_skips; i++) {
            seed = (seed * LCG_SKIP_2_32_MULTIPLIER + LCG_SKIP_2_32_ADDEND) & LCG_MASK;
        }
        dfz_initial_start -= large_skips * (1ULL << 32);

        // Medium skips (2^16)
        int medium_skips = floor((double)dfz_initial_start / (1ULL << 16));
        for (int i = 0; i < medium_skips; i++) {
            seed = (seed * LCG_SKIP_2_16_MULTIPLIER + LCG_SKIP_2_16_ADDEND) & LCG_MASK;
        }
        dfz_initial_start -= large_skips * (1ULL << 16);

        // Handle the rest of the lcg skips (up to 2^16 -1 iterations here)
        for (int i = 0; i < dfz_initial_start; i++) {
            seed = (seed * LCG_MULTIPLIER + LCG_ADDEND) & LCG_MASK;
        }
    }

    boinc_end_critical_section();


    // Block group being a group of cuda blocks
    int block_groups_needed = (total_blocks + BLOCKS_PER_GROUP - 1) / BLOCKS_PER_GROUP;
    if (DEBUG) {
        fprintf(stderr, "Doing %d total block groups, starting at %d, with %d blocks each containing %d threads.\n", block_groups_needed, block_group_start, BLOCKS_PER_GROUP, THREADS_PER_BLOCK);
    }
    time_t start_time = time(NULL);
    // unsigned long long seed = 112962399045242;  // Polymetrics test seed

    for (int block_group = block_group_start; block_group < block_groups_needed; block_group++) {

        // Minimize the extra work we are doing (yay efficiency)
        int blocks_to_do = BLOCKS_PER_GROUP;
        if ((block_group + 1) * BLOCKS_PER_GROUP > total_blocks) {
            blocks_to_do = (block_group + 1) * BLOCKS_PER_GROUP - total_blocks;
        }

        if (DEBUG) {
            fprintf(stderr, "%d %d %llu %llu\n", blocks_to_do, THREADS_PER_BLOCK, THREAD_WORK, TOTAL_WORK);
            fprintf(stderr, "[Block %d]: Searching %llu seeds.\n", block_group + 1, blocks_to_do * THREADS_PER_BLOCK * THREAD_WORK);
        }

        // Launch kernel
        if (DEBUG) {
            fprintf(stderr, "Starting seed: %llu.\n", seed);
        }
        gpu_bruteforce<<<blocks_to_do, THREADS_PER_BLOCK>>>(seed, counter, seed_list, missing_count_list, lcg_skip_list_multiplier, lcg_skip_list_addend);
        cudaDeviceSynchronize();

        boinc_begin_critical_section();


        time_t elapsed = time(NULL) - start_time;

        // Write block group to checkpoint
        fp = boinc_fopen("flower_checkpoint.txt", "w+");

        struct checkpoint_vars data_store;
        data_store.offset = block_group + 1;
        data_store.dfz_tracker = dfz_tracker;
        data_store.seed = seed;
        data_store.elapsed_chkpoint = elapsed_chkpoint + elapsed;
        data_store.total_counter = total_counter;

        fwrite(&data_store, sizeof(data_store), 1, fp);
        fclose(fp);

        boinc_checkpoint_completed();


        // Check outputs
        if (*counter > 0) {
            fp = boinc_fopen("output.txt", "a");
            total_counter += *counter;
            for (int i = 0; i < *counter; i++) {
                fprintf(fp, "Found seed. DFZ: %016llu, matches: %d.\n", dfz_tracker + seed_list[i], FLOWER_COUNT - missing_count_list[i]);
            }

            fclose(fp);
        }
        *counter = 0;

        // Boinc reporting
        boinc_fraction_done((double) block_group / block_groups_needed);

        boinc_end_critical_section();
    }

    

    boinc_begin_critical_section();

    // Calculate seeds per second (millions per second)
    time_t elapsed = time(NULL) - start_time;

    time_t total_time = elapsed_chkpoint + elapsed;

    double speed = (double) (TOTAL_WORK / 1000000) / total_time;
    
    printf("Found %llu valid seeds.\n", total_counter);
    printf("Finished in %llu seconds. Scanned %.2f million seeds per second.\n", (long long) total_time, speed);

    // Free memory
    cudaFree(counter);
    cudaFree(seed_list);

    // Delete checkpoint file
    boinc_delete_file("flower_checkpoint.txt");

    boinc_end_critical_section();
}



int main(int argc, char *argv[] ) {


    unsigned long long dfz = 0;
    int x = 0;
    int y = 0;
    int z = 0;


    // Parse arguments
    for (int i = 1; i < argc; i += 2) {
        const char *param = argv[i];
        if (strcmp(param, "--dfz") == 0) {
            dfz = atoi(argv[i + 1]);
        } else if (strcmp(param, "-x") == 0) {
            x = atoi(argv[i + 1]);
        } else if (strcmp(param, "-y") == 0) {
            y = atoi(argv[i + 1]);
        } else if (strcmp(param, "-z") == 0) {
            z = atoi(argv[i + 1]);
        }
    }


    center[0] = x;
    center[1] = y;
    center[2] = z;

    fprintf(stderr, "Received arguments: dfz: %llu, center: %d, %d, %d.\n", dfz, x, y, z);


    // Do the gpu thing
    run_kernel(dfz);
    fprintf(stderr, "Finished task!\n");

    boinc_finish(0);
}
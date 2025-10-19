#include "stencil5_H.h"


int main(int argc, char **argv)
{

  /* ==================================================================================================
  =                                                                                                   =
  =   Variables                                                                                       =  
  ===================================================================================================== */
  MPI_Comm myCOMM_WORLD;
  int  Rank, Ntasks;   // the rank of the calling process and the total number of MPI ranks (?)
  uint neighbours[4];  // in C an array passed to a function is passed as a pointer to the first element automatically
  int  Niterations;
  int  periodic;
  vec2_t S, N;   // size of the plate and the grid of MPI tasks
  int      Nsources;
  int      Nsources_local;
  vec2_t  *Sources_local;
  double   energy_per_source;
  plane_t   planes[2];  //2 planes, each one with attributes: size and pointer to data
  buffers_t buffers[2];  // result: double *restrict buffers[2][4] = 2 for send and receive, 4 for the neighbours (north, south, east, west)
  int output_energy_stat_perstep = 0;
  int debug_print_grid; //to debug, suggested ONLY FOR SMALL GRIDS, e.g. 8x8

  //seed for reproducibility
  long int seed;

  //time variables to profile the code
  double start1_time, start2_time, start3_time, start4_time, start5_time, start6_time, init_time, iter_time;
  double inject_energy_time = 0.0, MPIsetup_time = 0.0, inner_update_time = 0.0, border_update_time = 0.0, wait_time = 0.0, copy_buffers_time = 0.0;
  //double total_time;

  //debugging and statistics
  int times_statistics = 1;
  int verbose_times = 1; 
  int print_halo = 0;
  int print_full_grid_perstep = 0;

  //grid scan at initialization and update
  int scan_grid_by_blocks = 0; 

  
  /* ==================================================================================================
  =                                                                                                   =
  =   Initialize MPI environment                                                                      =  
  ===================================================================================================== */
  {
    int level_obtained;

    // NOTE: change MPI_FUNNELED if appropriate (with this the first thread is the only one allowed to make MPI calls)
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &level_obtained ); //checking if the specific MPI implementation offers this kind of thread support, otherwise we have an inferior level (corresponding to SINGLE). argc and argv are the same arguments passed to the main
    if ( level_obtained < MPI_THREAD_FUNNELED ) {
      printf("MPI_thread level obtained is %d instead of %d\n",
	     level_obtained, MPI_THREAD_FUNNELED );
      MPI_Finalize();
      exit(1); }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank); //obtain the rank of the calling process and save it in Rank
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks); 
    MPI_Comm_dup (MPI_COMM_WORLD, &myCOMM_WORLD); //copy the communicator MPI_COMM_WORLD to myCOMM_WORLD, so that we can use it in the rest of the code
  }
  

  /* ==================================================================================================
  =                                                                                                   =
  =   Arguments setting and checking and setting                                                      =  
  ===================================================================================================== */

  init_time = MPI_Wtime();
  int ret = initialize ( &myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep, &debug_print_grid,
			 neighbours, &Niterations,
			 &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
			 &planes[0], &buffers[0], &seed, &times_statistics, &verbose_times, &print_halo, &print_full_grid_perstep, &scan_grid_by_blocks
       );
  init_time = MPI_Wtime() - init_time; 


  /* Output messages: print actual grid scan mode after initialization */
  if (Rank == 0) {
    if (scan_grid_by_blocks) {
      printf("Datagrid scanned by blocks for initialization and update.\n");
    } else {
      printf("Datagrid scanned by rows for initialization and update.\n");
    }
  }
  

  if ( ret ) // if initialization failed = 1
    {
      printf("task %d is opting out with termination code %d\n", 
	     Rank, ret );
      
      MPI_Finalize();
      return 0;
    }
  
  


  /*---------------------------------------------------------------------------------------------------
  =====================================================================================================
  =                                                                                                   =
  =                                    GRID UPDATE ITERATIONS start                                   =
  =                                                                                                   =
  ===================================================================================================== */
  int current = OLD;
  iter_time = MPI_Wtime();   /* take wall-clock time */

  

  for (int iter = 0; iter < Niterations; ++iter)
    
    {
      /*handle for 8 asyncronous communication requests (4 neighbours, send and recv) */
      start1_time = MPI_Wtime();
      MPI_Request reqs[8];
      for (uint i = 0; i < 8; i++){
          reqs[i] = MPI_REQUEST_NULL; //to avoid undefined behaviour
      }

      
      /* new energy from sources */
      start2_time = MPI_Wtime();
      inject_energy( periodic, Nsources_local, (const vec2_t *)Sources_local, energy_per_source, &planes[current] , N);
      inject_energy_time += MPI_Wtime() - start2_time;

      
      // Print grids after energy injection
      if (print_halo && debug_print_grid && print_full_grid_perstep) {
          print_full_grid(myCOMM_WORLD, Ntasks, Rank, &N, &planes[current], print_halo, "OLD plane (after energy injection)", iter);
          print_full_grid(myCOMM_WORLD, Ntasks, Rank, &N, &planes[!current], print_halo, "NEW plane (after energy injection)", iter); 
      }





    /* ==================================================================================================
      =                                                                                                 =
      =   [A] fill the buffers, and/or make the buffers' pointers pointing to the correct position.     = 
      =   [B] perform the halo communications                                                           =
      =          (1) use Send / Recv (not implemented here)                                             =
      =          (2) use Isend / Irecv to overlap communication and computation                         = 
      =================================================================================================== */
      #define IDX( i, j ) ( (j)*(planes[current].size[_x_]+2)+ (i) )
      
      // North communications (tags are assigned from the receiver point of view)
      if (neighbours[NORTH] != MPI_PROC_NULL) {
          buffers[SEND][NORTH] = &planes[current].data[IDX(1, 1)];
          buffers[RECV][NORTH] = &planes[current].data[IDX(1, 0)];

          MPI_Isend( buffers[SEND][NORTH], planes[current].size[_x_], MPI_DOUBLE, neighbours[NORTH], SOUTH, myCOMM_WORLD, &reqs[SEND_NORTH]);
          MPI_Irecv( buffers[RECV][NORTH], planes[current].size[_x_], MPI_DOUBLE, neighbours[NORTH], NORTH, myCOMM_WORLD, &reqs[RECV_NORTH]);
      } 

      // South communications
      if (neighbours[SOUTH] != MPI_PROC_NULL) {
          buffers[SEND][SOUTH] = &planes[current].data[IDX(1, planes[current].size[_y_])];
          buffers[RECV][SOUTH] = &planes[current].data[IDX(1, planes[current].size[_y_]+1)];

          MPI_Isend( buffers[SEND][SOUTH], planes[current].size[_x_], MPI_DOUBLE, neighbours[SOUTH], NORTH, myCOMM_WORLD, &reqs[SEND_SOUTH]);
          MPI_Irecv( buffers[RECV][SOUTH], planes[current].size[_x_], MPI_DOUBLE, neighbours[SOUTH], SOUTH, myCOMM_WORLD, &reqs[RECV_SOUTH]);
      }

    
      // East communciations (east receviing buffers already set by memory_allocate. Nothing to point at, nothing to copy)
      if (neighbours[EAST] != MPI_PROC_NULL) {
          for ( uint j = 1; j <= planes[current].size[_y_]; j++ ) {
                buffers[SEND][EAST][j-1] = planes[current].data[IDX(planes[current].size[_x_], j)];
                }
          MPI_Isend( buffers[SEND][EAST], planes[current].size[_y_], MPI_DOUBLE, neighbours[EAST], WEST, myCOMM_WORLD, &reqs[SEND_EAST]);
          MPI_Irecv( buffers[RECV][EAST], planes[current].size[_y_], MPI_DOUBLE, neighbours[EAST], EAST, myCOMM_WORLD, &reqs[RECV_EAST]);
      }


      // West communications (west receviing buffers already set by memory_allocate. Nothing to point at, nothing to copy)
      if (neighbours[WEST] != MPI_PROC_NULL) {
          for ( uint j = 1; j <= planes[current].size[_y_]; j++ ) {
                    buffers[SEND][WEST][j-1] = planes[current].data[IDX(1, j)];
                }
          MPI_Isend( buffers[SEND][WEST], planes[current].size[_y_], MPI_DOUBLE, neighbours[WEST], EAST, myCOMM_WORLD, &reqs[SEND_WEST]);
          MPI_Irecv( buffers[RECV][WEST], planes[current].size[_y_], MPI_DOUBLE, neighbours[WEST], WEST, myCOMM_WORLD, &reqs[RECV_WEST]);
      }
      
      #undef IDX     
      MPIsetup_time += MPI_Wtime() - start1_time;



      
    /* ==================================================================================================
      =                                                                                                 =
      =   Check communications and update grid points                                                   = 
      =================================================================================================== */ 
      /*
      flags array to check completion status for each of the non-blocking MPI calls;
      (test_all_flag to check the completion of all the non-blocking MPI calls at the same time);
      (status to get more information about each of the non-blocking MPI calls).
      */ 
      int flags[8];
      //int test_all_flag;
      //MPI_Status status[8];


      start3_time = MPI_Wtime();
      update_plane( periodic, N, &planes[current], &planes[!current], UPDATE_INNER_GRID, &scan_grid_by_blocks); //inner grid update
      inner_update_time += MPI_Wtime() - start3_time;
        
      
      start4_time = MPI_Wtime();
      MPI_Waitall (8, reqs, MPI_STATUS_IGNORE); //wait for all the processes to have exchanged messages
      wait_time += MPI_Wtime() - start4_time;


      start5_time = MPI_Wtime();
      copy_buffers(flags, reqs, &buffers[0], &planes[current]);  // [C] copy the haloes data from RECEVING EAST AND WEST BUFFERS to the current plane
      copy_buffers_time += MPI_Wtime() - start5_time;


      start6_time = MPI_Wtime();
      update_borders( periodic, N, &planes[current], &planes[!current]);
      border_update_time += MPI_Wtime() - start6_time;

      // Print grids after complete update
      if (debug_print_grid && print_full_grid_perstep && (iter < Niterations - 1)) {
          print_full_grid(myCOMM_WORLD, Ntasks, Rank, &N, &planes[current], print_halo, "OLD plane (after update)", iter);
          print_full_grid(myCOMM_WORLD, Ntasks, Rank, &N, &planes[!current], print_halo, "NEW plane (after update)", iter);
      }
          



  /*  =================================================================================================
      =                                                                                               =
      =   Energy output and planes swap for the next iteration                                         = 
      ================================================================================================= */
      /* output if needed */
      if ( output_energy_stat_perstep )
	    output_energy_stat ( iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD );
	
      /* swap plane indexes for the new iteration */
      current = !current;
      
    }

  
  /* ==================================================================================================
  =                                                                                                   =
  =                                    GRID UPDATE ITERATIONS end                                     =
  =                                                                                                   =
  ===================================================================================================== 
  ----------------------------------------------------------------------------------------------------- */



  iter_time = MPI_Wtime() - iter_time;
  //total_time = init_time + iter_time;

  output_energy_stat ( -1, &planes[!current], Niterations * Nsources*energy_per_source, Rank, &myCOMM_WORLD );


  if (debug_print_grid) {
      print_full_grid(myCOMM_WORLD, Ntasks, Rank, &N, &planes[!current], print_halo, "FINAL NEW plane", -1);
  }






  /*  ==================================================================================================
   =                                                                                                   =
   =   TIME STATISTICS PRINTED ON TERMINAL                                                             =
   ==================================================================================================== */ 
  timing_data_t my_times = {
      init_time, iter_time, inject_energy_time, MPIsetup_time, 
      wait_time, inner_update_time, border_update_time, copy_buffers_time
  };

  if (times_statistics && Rank == 0) {
      timing_data_t all_times[Ntasks];
      
      MPI_Gather(&my_times, sizeof(timing_data_t), MPI_BYTE, 
                all_times, sizeof(timing_data_t), MPI_BYTE, 0, myCOMM_WORLD);
      
      // Find slowest rank based on iter_time
      int slowest_rank = 0;
      for (int i = 1; i < Ntasks; i++) {
          if (all_times[i].iter_time > all_times[slowest_rank].iter_time) {
              slowest_rank = i;
          }
      }
      
      if (verbose_times) {
      // Print slowest process details
      printf("\nSLOWEST PROCESS (seconds):\n");
      printf("-----------------------------------------------\n");
      printf("| Phase            |      Time     |\n");
      printf("-----------------------------------------------\n");
      printf("| Rank             | %12d    |\n", slowest_rank);
      printf("| Init             | %12.6f  |\n", all_times[slowest_rank].init_time);
      printf("| Iter             | %12.6f  |\n", all_times[slowest_rank].iter_time);
      printf("| Init+Iter        | %12.6f  |\n", all_times[slowest_rank].init_time + all_times[slowest_rank].iter_time);
      printf("| InjEn            | %12.6f  |\n", all_times[slowest_rank].inject_time);
      printf("| MPISetup         | %12.6f  |\n", all_times[slowest_rank].setup_time);
      printf("| Wait             | %12.6f  |\n", all_times[slowest_rank].wait_time);
      printf("| InnUpd           | %12.6f  |\n", all_times[slowest_rank].inner_time);
      printf("| BordUpd          | %12.6f  |\n", all_times[slowest_rank].border_time);
      printf("| CopyBuff         | %12.6f  |\n", all_times[slowest_rank].copy_time);
      printf("| SizeX            | %12u  |\n", S[_x_]);
      printf("| SizeY            | %12u  |\n", S[_y_]);
      printf("| GridX            | %12u  |\n", N[_x_]);
      printf("| GridY            | %12u  |\n", N[_y_]);
      printf("| Niter            | %12d  |\n", Niterations);
      printf("| Nsources         | %12d  |\n", Nsources);
      printf("| Periodic         | %12s  |\n", periodic ? "Yes" : "No");
      printf("-----------------------------------------------\n");
      } 

      // Compute statistics (if Ntasks > 1)
      double avg_init = all_times[0].init_time, min_init = all_times[0].init_time, max_init = all_times[0].init_time;
      double avg_iter = all_times[0].iter_time, min_iter = all_times[0].iter_time, max_iter = all_times[0].iter_time;
      double avg_inject = all_times[0].inject_time, min_inject = all_times[0].inject_time, max_inject = all_times[0].inject_time;
      double avg_setup = all_times[0].setup_time, min_setup = all_times[0].setup_time, max_setup = all_times[0].setup_time;
      double avg_wait = all_times[0].wait_time, min_wait = all_times[0].wait_time, max_wait = all_times[0].wait_time;
      double avg_inner = all_times[0].inner_time, min_inner = all_times[0].inner_time, max_inner = all_times[0].inner_time;
      double avg_border = all_times[0].border_time, min_border = all_times[0].border_time, max_border = all_times[0].border_time;
      double avg_copy = all_times[0].copy_time, min_copy = all_times[0].copy_time, max_copy = all_times[0].copy_time;
      
      for (int i = 1; i < Ntasks; i++) {
          // Sum for averages
          avg_init += all_times[i].init_time;
          avg_iter += all_times[i].iter_time;
          avg_inject += all_times[i].inject_time;
          avg_setup += all_times[i].setup_time;
          avg_wait += all_times[i].wait_time;
          avg_inner += all_times[i].inner_time;
          avg_border += all_times[i].border_time;
          avg_copy += all_times[i].copy_time;
          
          // Min/Max
          if (all_times[i].init_time < min_init) min_init = all_times[i].init_time;
          if (all_times[i].init_time > max_init) max_init = all_times[i].init_time;
          if (all_times[i].iter_time < min_iter) min_iter = all_times[i].iter_time;
          if (all_times[i].iter_time > max_iter) max_iter = all_times[i].iter_time;
          if (all_times[i].inject_time < min_inject) min_inject = all_times[i].inject_time;
          if (all_times[i].inject_time > max_inject) max_inject = all_times[i].inject_time;
          if (all_times[i].setup_time < min_setup) min_setup = all_times[i].setup_time;
          if (all_times[i].setup_time > max_setup) max_setup = all_times[i].setup_time;
          if (all_times[i].wait_time < min_wait) min_wait = all_times[i].wait_time;
          if (all_times[i].wait_time > max_wait) max_wait = all_times[i].wait_time;
          if (all_times[i].inner_time < min_inner) min_inner = all_times[i].inner_time;
          if (all_times[i].inner_time > max_inner) max_inner = all_times[i].inner_time;
          if (all_times[i].border_time < min_border) min_border = all_times[i].border_time;
          if (all_times[i].border_time > max_border) max_border = all_times[i].border_time;
          if (all_times[i].copy_time < min_copy) min_copy = all_times[i].copy_time;
          if (all_times[i].copy_time > max_copy) max_copy = all_times[i].copy_time;
      }
      
      // Averages
      avg_init /= Ntasks; avg_iter /= Ntasks; avg_inject /= Ntasks; avg_setup /= Ntasks;
      avg_wait /= Ntasks; avg_inner /= Ntasks; avg_border /= Ntasks; avg_copy /= Ntasks;
      
      if (Ntasks > 1 && verbose_times) {
      printf("\nTIME STATISTICS (seconds):\n");
      printf("---------------------------------------------------------------------------------\n");
      printf("| Phase            |   Avg Time |   Min Time |   Max Time |\n");
      printf("---------------------------------------------------------------------------------\n");
      printf("| Initialization   | %10.6f | %10.6f | %10.6f |\n", avg_init, min_init, max_init);
      printf("| Total Iterations | %10.6f | %10.6f | %10.6f |\n", avg_iter, min_iter, max_iter);
      printf("| Inject Energy    | %10.6f | %10.6f | %10.6f |\n", avg_inject, min_inject, max_inject);
      printf("| MPI Setup        | %10.6f | %10.6f | %10.6f |\n", avg_setup, min_setup, max_setup);
      printf("| Wait             | %10.6f | %10.6f | %10.6f |\n", avg_wait, min_wait, max_wait);
      printf("| Inner Update     | %10.6f | %10.6f | %10.6f |\n", avg_inner, min_inner, max_inner);
      printf("| Border Update    | %10.6f | %10.6f | %10.6f |\n", avg_border, min_border, max_border);
      printf("| Copy Buffers     | %10.6f | %10.6f | %10.6f |\n", avg_copy, min_copy, max_copy);
      printf("---------------------------------------------------------------------------------\n");
      }
      
      // Save in arrays for CSV output
      double all_init_times[Ntasks], all_iter_times[Ntasks], all_inject_times[Ntasks], all_setup_times[Ntasks];
      double all_wait_times[Ntasks], all_inner_times[Ntasks], all_border_times[Ntasks], all_copy_times[Ntasks];
      
      for (int i = 0; i < Ntasks; i++) {
          all_init_times[i] = all_times[i].init_time;
          all_iter_times[i] = all_times[i].iter_time;
          all_inject_times[i] = all_times[i].inject_time;
          all_setup_times[i] = all_times[i].setup_time;
          all_wait_times[i] = all_times[i].wait_time;
          all_inner_times[i] = all_times[i].inner_time;
          all_border_times[i] = all_times[i].border_time;
          all_copy_times[i] = all_times[i].copy_time;
      }
         
      // CSV outputs
      printf("Writing CSV files...\n");

      
      /*==============================================================================================
      =                                                                                             =
      =  CSV FILES WITH TIME STATISTICS. Printed in the executable directory                        =                                                                                 =
      =============================================================================================== */ 
      // FILE 1: slowest rank details
      const char *path1 = "slowest_rank_times.csv";
      int has_header1 = file_exists_and_nonempty(path1);
      
      FILE *fp1 = fopen(path1, "a");
      if (fp1) {
          // Header if necessary
          if (!has_header1) {
              fprintf(fp1, "GridScan,RunTS,SLURM_JOB_ID,NODES,MPI_RANKS,OMP_THREADS,OMP_PLACES,OMP_PROC_BIND,"
                          "SizeX,SizeY,GridX,GridY,Niter,Nsources,Periodic,"
                          "Rank,Total,Init,Iter,Inject,MPIsetup,Wait,InnerUpdate,BorderUpdate,Update,CopyBuffers\n");
          }

          const char *jobid = getenv("SLURM_JOB_ID");
          const char *env_nodes = getenv("SLURM_JOB_NUM_NODES");
          const char *omp_places = getenv("OMP_PLACES");
          const char *omp_proc_bind = getenv("OMP_PROC_BIND");
          char ts[32]; 
          time_t t = time(NULL);
          struct tm tmv;
          gmtime_r(&t, &tmv);
          strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);

          int omp_threads = 1;
          #ifdef _OPENMP //activates at compilation time if the flag -fopenmp is used
          omp_threads = omp_get_max_threads();
          #endif
          double total_slowest = all_init_times[slowest_rank] + all_iter_times[slowest_rank];
          double update_slowest = all_inner_times[slowest_rank] + all_border_times[slowest_rank];

          fprintf(fp1, "%s,%s,%s,%s,%d,%d,%s,%s,"
                      "%u,%u,%u,%u,%d,%d,%s,"
                      "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                  scan_grid_by_blocks ? "ByBlocks" : "ByRows",
                  ts,
                  jobid ? jobid : "(unset)",
                  env_nodes ? env_nodes : "(unset)",
                  Ntasks,
                  omp_threads,
                  omp_places ? omp_places : "(unset)",
                  omp_proc_bind ? omp_proc_bind : "(unset)",
                  S[_x_], S[_y_], 
                  N[_x_], N[_y_], 
                  Niterations, 
                  Nsources,
                  periodic ? "Yes" : "No",
                  slowest_rank, 
                  total_slowest,
                  all_init_times[slowest_rank], 
                  all_iter_times[slowest_rank],
                  all_inject_times[slowest_rank], 
                  all_setup_times[slowest_rank],
                  all_wait_times[slowest_rank], 
                  all_inner_times[slowest_rank],
                  all_border_times[slowest_rank], 
                  update_slowest, 
                  all_copy_times[slowest_rank]
          );
          fclose(fp1);
      }

      // FILE 2: Aggregated statistics
      if (Ntasks > 1) {
          const char *path2 = "stats_avg_min_max.csv";
          int has_header2 = file_exists_and_nonempty(path2);

      FILE *fp2 = fopen(path2, "a");
      if (fp2) {
          // Header if necessary
          if (!has_header2) {
              fprintf(fp2, "GridScan,RunTS,SLURM_JOB_ID,NODES,MPI_RANKS,OMP_THREADS,OMP_PLACES,OMP_PROC_BIND,"
                          "SizeX,SizeY,GridX,GridY,Niter,Nsources,Periodic,"
                          "Phase,SlowestRank,Avg,Min,Max\n");
          }

          const char *jobid = getenv("SLURM_JOB_ID");
          const char *env_nodes = getenv("SLURM_JOB_NUM_NODES");
          const char *omp_places = getenv("OMP_PLACES");
          const char *omp_proc_bind = getenv("OMP_PROC_BIND");
          char ts[32];
          time_t t = time(NULL);
          struct tm tmv;
          gmtime_r(&t, &tmv);
          strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);

          int omp_threads = 1;
          #ifdef _OPENMP
          omp_threads = omp_get_max_threads();
          #endif

          struct {
              const char *phase;
              double slowest, avg, min, max;
          } stats[] = {
              {"Initialization", all_init_times[slowest_rank], avg_init, min_init, max_init},
              {"Total Iterations", all_iter_times[slowest_rank], avg_iter, min_iter, max_iter},
              {"Inject Energy", all_inject_times[slowest_rank], avg_inject, min_inject, max_inject},
              {"MPI Setup", all_setup_times[slowest_rank], avg_setup, min_setup, max_setup},
              {"Wait", all_wait_times[slowest_rank], avg_wait, min_wait, max_wait},
              {"Inner Update", all_inner_times[slowest_rank], avg_inner, min_inner, max_inner},
              {"Border Update", all_border_times[slowest_rank], avg_border, min_border, max_border},
              {"Update", all_inner_times[slowest_rank] + all_border_times[slowest_rank],
                        avg_inner + avg_border, min_inner + min_border, max_inner + max_border},
              {"Copy Buffers", all_copy_times[slowest_rank], avg_copy, min_copy, max_copy}
          };

          for (int i = 0; i < sizeof(stats)/sizeof(stats[0]); ++i) {
              fprintf(fp2, "%s,%s,%s,%s,%d,%d,%s,%s,"
                          "%u,%u,%u,%u,%d,%d,%s,"
                          "%s,%.6f,%.6f,%.6f,%.6f\n",
                  scan_grid_by_blocks ? "ByBlocks" : "ByRows",
                  ts,
                  jobid ? jobid : "(unset)",
                  env_nodes ? env_nodes : "(unset)",
                  Ntasks,
                  omp_threads,
                  omp_places ? omp_places : "(unset)",
                  omp_proc_bind ? omp_proc_bind : "(unset)",
                  S[_x_], S[_y_], 
                  N[_x_], N[_y_], 
                  Niterations, 
                  Nsources,
                  periodic ? "Yes" : "No",
                  stats[i].phase, 
                  stats[i].slowest, 
                  stats[i].avg, 
                  stats[i].min, 
                  stats[i].max
              );
          }
          fclose(fp2);
      }
      }

      printf("CSV files written successfully.\n");

  } else if (times_statistics) {
      MPI_Gather(&my_times, sizeof(timing_data_t), MPI_BYTE, NULL, 0, MPI_BYTE, 0, myCOMM_WORLD);
  }






  /*===========================================================================
   =                                                                          =
   =   memory release                                                         =
   ============================================================================*/
  memory_release( planes, buffers ); 

  MPI_Finalize();
  
  return 0;
}


















/* ==================================================================================================
=                                                                                                   =
=                                       FUNCTION DEFINITIONS                                        =
=                                                                                                   =
===================================================================================================== */

int initialize ( 
     MPI_Comm *Comm,
		 int      Me,                  // the rank of the calling process
		 int      Ntasks,              // the total number of MPI ranks
		 int      argc,                // the argc from command line
		 char   **argv,                // the argv from command line
		 vec2_t  *S,                   // the size of the plane, since this is the require argument S has to be passed as &S (pointer to the whole object, C knows is made of 2 elements). Passing S would mean otherwise vect_2 S = uint * S (pointer to the first element of the vector)
		 vec2_t  *N,                   // two-uint array defining the MPI tasks' grid
		 int     *periodic,            // periodic-boundary tag
		 int     *output_energy_stat,
     int     *debug_print_grid,
		 uint     *neighbours,          // four-int array that gives back the neighbours of the calling task
		 int     *Niterations,         // how many iterations
		 int     *Nsources,            // how many heat sources
		 int     *Nsources_local,
		 vec2_t  **Sources_local,
		 double  *energy_per_source,   // how much heat per source
		 plane_t *planes,
		 buffers_t *buffers,
     long int *seed,
     int *verbose_times,
     int *time_statistics,
     int *print_halo,
     int *print_full_grid_perstep,
     int *scan_grid_by_blocks
		 )
{
  int halt = 0;
  int ret; 
  int verbose = 0;

  // ··································································
  // set deffault values

  (*S)[_x_]         = 10000;
  (*S)[_y_]         = 10000;
  *periodic         = 0;
  *Nsources         = 4;
  *Nsources_local   = 0;
  *Sources_local    = NULL;
  *Niterations      = 1000;
  *energy_per_source = 1.0;
  *debug_print_grid = 0; //to debug, suggested ONLY FOR SMALL GRIDS, e.g. 8x8
  *seed = 12345;

  if ( planes == NULL ) {
    if (Me == 0) printf("Error: two data planes must be allocated\n");
    return 1;
  }

  planes[OLD].size[_x_] = planes[OLD].size[_y_] = 0;  
  planes[NEW].size[_x_] = planes[NEW].size[_y_] = 0;

  for ( int i = 0; i < 4; i++ )
    neighbours[i] = MPI_PROC_NULL;

  for ( int b = 0; b < 2; b++ ) //two buffers: to send and receive
    for ( int d = 0; d < 4; d++ ) // 4 for the neighbours (north, south, east, west)
      buffers[b][d] = NULL;
  
  //--------------------------------------------------------------------------
  //environment variables for general setting of the code
  const char *env = getenv("TIMES_STATISTICS");
  if (env != NULL) {
    int val = atoi(env);
    if (!(val == 0 || val == 1)) {
      fprintf(stderr, "Error: TIMES_STATISTICS must be either 0 or 1 ('%s'). Set to 0.\n", env);
      val = 0;
    }
    *time_statistics = val;
    *verbose_times = val;
  }

  const char *grid_mode = getenv("GRID_PRINT_MODE");
  if (grid_mode != NULL) {
    if (strcmp(grid_mode, "Debug") == 0) {
      *debug_print_grid = 1;
      *print_halo = 1;
      *print_full_grid_perstep = 1;
    } else if (strcmp(grid_mode, "GridEachStep") == 0) {
      *debug_print_grid = 1;
      *print_halo = 0;
      *print_full_grid_perstep = 1;
    } else if (strcmp(grid_mode, "GridFinal") == 0) {
      *debug_print_grid = 1;
      *print_halo = 0;
      *print_full_grid_perstep = 0;
    } else {
      fprintf(stderr, "Error: GRID_PRINT_MODE must be 'Debug', 'GridEachStep' or 'GridFinal' (got '%s'). Grid printing disabled.\n", grid_mode);
      *debug_print_grid = 0;
      *print_halo = 0;
      *print_full_grid_perstep = 0;
    }
  }

  // ··································································
  // process the command line (CLI parameters have precedence over env variables)
  while ( 1 )
  {
    int opt;
    while((opt = getopt(argc, argv, ":hx:y:e:E:n:o:p:v:D:P:d:T:t:l:s:")) != -1) //getopt returns -1 when all options have been processed
      {
	switch( opt )
	  {
	  case 'x': (*S)[_x_] = (uint)atoi(optarg); //atoi to go from string "123"(e.g) to integer 123(e.g)
	    break;

	  case 'y': (*S)[_y_] = (uint)atoi(optarg);
	    break;

	  case 'e': *Nsources = atoi(optarg);
	    break;

	  case 'E': *energy_per_source = atof(optarg);
	    break;

	  case 'n': *Niterations = atoi(optarg);
	    break;

	  case 'o': *output_energy_stat = (atoi(optarg) > 0); //se maggiore di 0 viene impostato a 1
	    break;

	  case 'p': *periodic = (atoi(optarg) > 0);
	    break;

	  case 'v': verbose = atoi(optarg);
	    break;

    case 'D': *debug_print_grid = (atoi(optarg) > 0); //to debug, suggested ONLY FOR SMALL GRIDS, e.g. 8x8
      break;

    case 'P': *print_full_grid_perstep = (atoi(optarg) > 0);
      break;

    case 'd': *print_halo = (atoi(optarg) > 0); //to print halo info at each iteration
      break;

    case 'T': *verbose_times = (atoi(optarg) > 0);
      break;  

    case 't': *time_statistics = (atoi(optarg) > 0);
      break;

    case 'l': *scan_grid_by_blocks = (atoi(optarg) > 0);
      break;

	  case 'h': {
	    if ( Me == 0 )
	      printf( "\nvalid options are ( values btw [] are the default values ):\n"
		      "-x    x size of the plate [10000]\n"
		      "-y    y size of the plate [10000]\n"
		      "-e    how many energy sources on the plate [4]\n"
		      "-E    how many energy sources on the plate [1.0]\n"
		      "-n    how many iterations [1000]\n"
		      "-p    whether periodic boundaries applies  [0 = false]\n\n"
          "-o    whether to output energy statistics at each step [0 = false]\n"
          "-D    whether to print the grid at each step for debugging purposes [0 = false]. Use with small grids only \n"
          "-d    whether to print the grid with halo information for debugging purposes [0 = false]\n"
          "-s    seed for random number generator [12345]\n"
          "-v    verbosity level [0]\n"
          "-h    this help\n\n"
          "-T   whether to stdout print verbose timing statistics at the end [0 = false]\n"
          "-t   whether to compute and output timing statistics at the end in csv files [0 = false]\n"
          "-P   whether to print the full grid at each step [0 = false]\n"
          "-l   whether to scan the grid by blocks when printing. Default is row-major scanning[0 = false], block scanning[1 = true]\n"
		      );
	    halt = 1; }
	    break;

    case 's': *seed = atol(optarg); 
      break;
	    
	  case ':': printf( "option -%c requires an argument\n", optopt);
	    break;
	    
	  case '?': printf(" -------- help unavailable ----------\n");
	    break;
	  }
      }

    if ( opt == -1 )
      break;
  }

  if ( halt )
    return 1;
  
  
  // ··································································
  /* here we check for all the parms being meaningful */
  if ((*S)[_x_] <= 0 || (*S)[_y_] <= 0) {
    if (Me == 0) printf("Error: Plate size must be positive.\n");
    return 1;
  }
  if (*Niterations <= 0) {
    if (Me == 0) printf("Error: Number of iterations must be positive.\n");
    return 1;
  }
  if (*Nsources < 0) {
    if (Me == 0) printf("Error: Number of sources cannot be negative.\n");
    return 1;
  }
  if (*energy_per_source < 0) {
    if (Me == 0) printf("Error: Energy per source cannot be negative.\n");
    return 1;
  }
  if (Ntasks <= 0) {
    if (Me == 0) printf("Error: Number of MPI tasks must be positive.\n");
    return 1;
  }
  if (Ntasks > ((*S)[_x_] * (*S)[_y_])) {
    if (Me == 0) {
      printf("Warning: Number of MPI tasks is greater than the number of grid points.\n");
      printf("Each task may be assigned zero or one grid point. This can lead to inefficient resource usage or idle tasks.\n");
    return 1;
    }
  }
  
  // ··································································
  /* Very simple algorithm for domain decomposition, you may want to
   * substitute it with a better one
   *
   * the plane Sx x Sy will be solved with a grid
   * of Nx x Ny MPI tasks
   */

  vec2_t Grid;  //dimensions of the grid of MPI tasks
  double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_]/(*S)[_y_] : (double)(*S)[_y_]/(*S)[_x_] );  //*S contains the sizes of the plate asked by the user
  /*
   * above formfactor is double for precise computations. 
   * the cast from double to int always keeps the integer part only, 
   * we add 1 to consider the smallest number greater than the formfactor.
   */
  int    dimensions = 2 - (Ntasks <= ((int)formfactor+1) );  //when Ntasks <= ((int)formfactor+1) is true, the grid is 1D because the number of tasks is too small to fill a 2D grid



  // depending on the grid choosen shape and the number of tasks, the domain is decomposed to avoid to narrow domains and to have few neighbours when possible (grid 1XN or viceversa)
  if ( dimensions == 1 ) // only if Ntasks <= ((int)formfactor+1) is true
    {
      if ( (*S)[_x_] >= (*S)[_y_] )
        Grid[_x_] = Ntasks, Grid[_y_] = 1;
      else
        Grid[_x_] = 1, Grid[_y_] = Ntasks;
    }
  else
    { 
      int   Nf;
      uint *factors;
      uint  first = 1;
      ret = simple_factorization( Ntasks, &Nf, &factors ); //of the chosen Ntask
      if (ret != 0) {
        if (Me == 0) printf("Error: simple_factorization failed.\n");
        return 1;
      }
      for ( int i = 0; (i < Nf) && ((Ntasks/first)/first > formfactor); i++ )
        first *= factors[i];

      if ( (*S)[_x_] > (*S)[_y_] )
        Grid[_x_] = Ntasks/first, Grid[_y_] = first;
      else
        Grid[_x_] = first, Grid[_y_] = Ntasks/first;
    }

  // dimensions of the grid of MPI tasks
  (*N)[_x_] = Grid[_x_]; 
  (*N)[_y_] = Grid[_y_];
  

  // ··································································
  // my cooridnates in the grid of processors
  //
  int X = Me % Grid[_x_]; //output the remainder of the division Me/Grid[_x_]
  int Y = Me / Grid[_x_]; //output the integer division of Me/Grid[_x_]

  // ··································································
  // find my neighbours
  //

  if ( Grid[_x_] > 1 )
    {  
      if ( *periodic ) {       
	neighbours[EAST]  = Y*Grid[_x_] + (Me + 1 ) % Grid[_x_];
	neighbours[WEST]  = (X%Grid[_x_] > 0 ? Me-1 : (Y+1)*Grid[_x_]-1); }
      
      else {
	neighbours[EAST]  = ( X < Grid[_x_]-1 ? Me+1 : MPI_PROC_NULL );
	neighbours[WEST]  = ( X > 0 ? (Me-1)%Ntasks : MPI_PROC_NULL ); }  
    }

  if ( Grid[_y_] > 1 )
    {
      if ( *periodic ) {      
	neighbours[NORTH] = (Ntasks + Me - Grid[_x_]) % Ntasks;
	neighbours[SOUTH] = (Ntasks + Me + Grid[_x_]) % Ntasks; }

      else {    
	neighbours[NORTH] = ( Y > 0 ? Me - Grid[_x_]: MPI_PROC_NULL );
	neighbours[SOUTH] = ( Y < Grid[_y_]-1 ? Me + Grid[_x_] : MPI_PROC_NULL ); }
    }


  // ··································································
  // the size of my patch
  //

  /*
   * every MPI task determines the size sx x sy of its own domain
   * REMIND: the computational domain will be embedded into a frame
   *         that is (sx+2) x (sy+2)
   *         the outern frame will be used for halo communication or
   */
  
  vec2_t mysize;
  uint s = (*S)[_x_] / Grid[_x_];
  uint r = (*S)[_x_] % Grid[_x_];  //r is for remainder
  mysize[_x_] = s + (X < r); //X was the column coordinate in the MPI grid. We distribute the remainder among the first r tasks in the row
  s = (*S)[_y_] / Grid[_y_];
  r = (*S)[_y_] % Grid[_y_];
  mysize[_y_] = s + (Y < r);

  planes[OLD].size[0] = mysize[0];
  planes[OLD].size[1] = mysize[1];
  planes[NEW].size[0] = mysize[0];
  planes[NEW].size[1] = mysize[1];
  
  // print the grid decomposition if verbose is enabled
  if ( verbose > 0 )
    {
          if ( Me == 0 ) {
      printf("Tasks are decomposed in a grid %d x %d\n\n",
        Grid[_x_], Grid[_y_] );
      fflush(stdout);
          }

          MPI_Barrier(*Comm);
          
          for ( int t = 0; t < Ntasks; t++ )
      {
        if ( t == Me )
          {
            printf("Task %4d :: "
            "\tgrid coordinates : %3d, %3d\n"
            "\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
            Me, X, Y,
            neighbours[NORTH], neighbours[EAST],
            neighbours[SOUTH], neighbours[WEST] );
            fflush(stdout);
          }

        MPI_Barrier(*Comm);
      }
      
    }

  
  // ··································································
  // allocate the needed memory
  //
  ret = memory_allocate( (const int *)neighbours, buffers , planes, scan_grid_by_blocks);
  if (ret != 0) {
    if (Me == 0) printf("Error: memory_allocate failed.\n");
    return 1;
  }

  // ··································································
  // allocate the heat sources
  //
  ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local, seed);
  if (ret != 0) {
    if (Me == 0) printf("Error: initialize_sources failed.\n");
    return 1;
  }

  return 0;
}


uint simple_factorization( uint A, int *Nfactors, uint **factors ) 
/*
 * rought factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
 #
 */
{
  //initialize the number of factors
  int N = 0;
  int f = 2;
  uint _A_ = A;

  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	N++;
	_A_ /= f; }  

      f++;
    }

  *Nfactors = N;
  uint *_factors_ = (uint*)malloc( N * sizeof(uint) );   

  //store the factors
  N   = 0;
  f   = 2;
  _A_ = A;

  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	_factors_[N++] = f;
	_A_ /= f; } 
      f++;
    }

  *factors = _factors_;
  return 0;
}


int initialize_sources(
      int       Me,
			int       Ntasks,
			MPI_Comm *Comm,
			vec2_t    mysize,
			int       Nsources,
			int      *Nsources_local,
			vec2_t  **Sources, 
      long int *seed)
{

  srand48((*seed) ^ Me);  //different seed for each MPI task and iteration. XOR is a bitwise operation, compares bits of the arguments.
  int *tasks_with_sources = (int*)malloc( Nsources * sizeof(int) ); //malloc allocates a block of VIRTUAL memory of Nsources integers, that will be used to store the tasks that will have sources. Output a pointer. ((The OS allocates physical memory only when data are accessed via reading or writing))
  
  if ( Me == 0 )
    {
      for ( int i = 0; i < Nsources; i++ )
	tasks_with_sources[i] = (int)lrand48() % Ntasks; //starting with a random number, we devide and take the remainder (to be sure that the number is in the range 0..Ntasks-1)
    }
  
  MPI_Bcast( tasks_with_sources, Nsources, MPI_INT, 0, *Comm ); //send all tasks the vector of sources = Nsources integer elements (MPI knows the data type of elements but not the size of the vector). 0 is the sending root, the receivers are all the other tasks regarding the same communicator.

  int nlocal = 0;
  for ( int i = 0; i < Nsources; i++ )        
    nlocal += (tasks_with_sources[i] == Me); //count how many sources are in the local task
  *Nsources_local = nlocal;
  
  if ( nlocal > 0 )
    {
      vec2_t * restrict helper = (vec2_t*)malloc( nlocal * sizeof(vec2_t) );     //an array of coordinates for each source in nlocal !!! so task-specific
      for ( int s = 0; s < nlocal; s++ )
	{
    //keep in mind we start from the coordinates 1,1 because the first row and column are used for halo communication
	  helper[s][_x_] = 1 + lrand48() % mysize[_x_]; //first access at the nlocal[i] element and then get/set a certain coordinate (RANDOM)
	  helper[s][_y_] = 1 + lrand48() % mysize[_y_];
	}

      *Sources = helper;
    }
  
  free( tasks_with_sources ); // we don't need this anymore, each process now should have its own sources

  return 0;
}



int memory_allocate (const int *neighbours, buffers_t *buffers_ptr, plane_t *planes_ptr, int *scan_grid_by_blocks )
{
  if (planes_ptr == NULL ) {
      fprintf(stderr, "Error: planes_ptr is NULL in memory_allocate.\n");
      return 1;
  }


  if (buffers_ptr == NULL ) {
      fprintf(stderr, "Error: buffers_ptr is NULL in memory_allocate.\n");
      return 1;
  }
    

  // ··················································
  // allocate memory for data
  // we allocate the space needed for the plane plus a contour frame that will contains data form neighbouring MPI tasks
  unsigned int frame_size = (planes_ptr[OLD].size[_x_]+2) * (planes_ptr[OLD].size[_y_]+2);

  planes_ptr[OLD].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[OLD].data == NULL ){
    fprintf(stderr, "Error: malloc failed for planes_ptr[OLD].data in memory_allocate.\n");
    return 1;
  }


  planes_ptr[NEW].data = (double*)malloc( frame_size * sizeof(double) );
  if ( planes_ptr[NEW].data == NULL ){
    fprintf(stderr, "Error: malloc failed for planes_ptr[NEW].data in memory_allocate.\n");
    return 1;
  }



  // ··················································
  // Parallel initialization: by rows or by blocks
  uint fysize = planes_ptr[OLD].size[_y_]+2;
  uint fxsize = planes_ptr[OLD].size[_x_]+2;

  #define IDX(i,j) ((j)*fxsize + (i))


  if (!(*scan_grid_by_blocks)) {

    // Row-wise scan
    #pragma omp parallel for schedule(static)
    for (uint j = 0; j < fysize; j++)
      for (uint i = 0; i < fxsize; i++) {
        planes_ptr[OLD].data[IDX(i,j)] = 0.0;
        planes_ptr[NEW].data[IDX(i,j)] = 0.0;
      }

  } else {

    // Block-wise scan
    const int block_size_x = 64;
    const int block_size_y = 48;

    #pragma omp parallel for collapse(2) schedule(static) //parallelize the two outer loops
    for (uint j = 0; j < fysize; j += block_size_y)
      for (uint i = 0; i < fxsize; i += block_size_x)
        for (uint jj = j; jj < j + block_size_y && jj < fysize; jj++)
          for (uint ii = i; ii < i + block_size_x && ii < fxsize; ii++) {
            planes_ptr[OLD].data[IDX(ii,jj)] = 0.0;
            planes_ptr[NEW].data[IDX(ii,jj)] = 0.0;
          }
  }

  #undef IDX



  // ··················································
  // allocate buffers
  /* buffers for north and south communication 
  are not really needed
  
  in fact, they are already contiguous, just the
  first and last line of every rank's plane
  
  you may just make some pointers pointing to the
  correct positions (beginning of first and last row of the OLD plane)
  

  or, if you prefer, just go on and allocate buffers
  also for north and south communications */
  if (neighbours[EAST] != MPI_PROC_NULL) {
    buffers_ptr[SEND][EAST]  = (double*)malloc( planes_ptr[OLD].size[_y_] * sizeof(double) ); 
    buffers_ptr[RECV][EAST]  = (double*)malloc( planes_ptr[OLD].size[_y_] * sizeof(double) );
  } else {
    buffers_ptr[SEND][EAST]  = NULL;
    buffers_ptr[RECV][EAST]  = NULL;
  }

  if (neighbours[WEST] != MPI_PROC_NULL) {
    buffers_ptr[SEND][WEST]  = (double*)malloc( planes_ptr[OLD].size[_y_] * sizeof(double) );
    buffers_ptr[RECV][WEST]  = (double*)malloc( planes_ptr[OLD].size[_y_] * sizeof(double) );
  } else {
    buffers_ptr[SEND][WEST]  = NULL;
    buffers_ptr[RECV][WEST]  = NULL;
  }

  //pointers are defined in  ***where***, we do not really allocate memory for the north and south buffers
  buffers_ptr[SEND][NORTH] = NULL;
  buffers_ptr[SEND][SOUTH] = NULL; 
  buffers_ptr[RECV][NORTH] = NULL;
  buffers_ptr[RECV][SOUTH] = NULL;

  // Check for malloc failure
  if ((neighbours[EAST] != MPI_PROC_NULL && (!buffers_ptr[SEND][EAST] || !buffers_ptr[RECV][EAST])) ||
      (neighbours[WEST] != MPI_PROC_NULL && (!buffers_ptr[SEND][WEST] || !buffers_ptr[RECV][WEST]))) {
      fprintf(stderr, "Error: malloc failed for communication buffers (EAST/WEST) in memory_allocate.\n");
      return -1;
  }
  
  return 0;
}



int memory_release ( plane_t *planes, buffers_t *buffers)
{
  if (planes != NULL) {
      free(planes[OLD].data); planes[OLD].data = NULL;
      free(planes[NEW].data); planes[NEW].data = NULL;
  }

  if (buffers != NULL) {
    for (int direction = 0; direction < 2; direction++) { // SEND and RECV
      
        free(buffers[direction][EAST]); buffers[direction][EAST] = NULL;
        free(buffers[direction][WEST]); buffers[direction][WEST] = NULL;
        buffers[direction][NORTH] = NULL; //no memory allocated
        buffers[direction][SOUTH] = NULL; //no memory allocated
    }
  }

  return 0;
}



int output_energy_stat (int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm ) 
{
  double system_energy = 0;
  double tot_system_energy = 0;
  get_total_energy ( plane, &system_energy ); //little remark: in C a pointer passed to a function remains a pointer, a primitive data type is passed by value instead , so the value is copied in the function and the original value is not modified (using & we are addressing the original system_energy variable, defined at the beginning of the main)
  
  MPI_Reduce ( &system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm );  
 
  if ( Me == 0 )
    {
      if ( step >= 0 ) {
	printf(" [ step %4d ] ", step ); fflush(stdout);
      }

      printf( "total injected energy is %g, "
	      "system energy is %g "
	      "( in avg %g per grid point)\n",
	      budget,
	      tot_system_energy,
	      tot_system_energy / (plane->size[_x_]*plane->size[_y_]) );
    }
  
  return 0;
}



inline int copy_buffers(int *flags, MPI_Request *reqs, buffers_t *buffers, plane_t *plane)
{
    MPI_Test(&reqs[RECV_EAST], &flags[RECV_EAST], MPI_STATUS_IGNORE); //receiving from EAST neighbour
    #define IDX(i,j) ((j)*(plane->size[_x_]+2) + (i))
    if (flags[RECV_EAST] && buffers[RECV][EAST] != NULL) {
        for (int k = 0; k < plane->size[_y_]; k++) {
            plane->data[IDX(plane->size[_x_]+1, k+1)] = buffers[RECV][EAST][k];
        }
    }

    MPI_Test(&reqs[RECV_WEST], &flags[RECV_WEST], MPI_STATUS_IGNORE); //receiving from WEST neighbour
    if (flags[RECV_WEST] && buffers[RECV][WEST] != NULL) {
        for (int k = 0; k < plane->size[_y_]; k++) {
            plane->data[IDX(0, k+1)] = buffers[RECV][WEST][k];
        }
    }
    #undef IDX

    return 0;
}


// Print the full grid (all subgrids) in correct MPI grid order, only on rank 0
// If print_halo is true, print also halos and highlight them
int print_full_grid(MPI_Comm comm, int Ntasks, int Rank, vec2_t *Grid, plane_t *plane, int print_halo, const char *plane_label, int iter) 
{
  int sx = plane->size[_x_];
  int sy = plane->size[_y_];
  int nx = sx + 2;
  int ny = sy + 2;
  int local_size = print_halo ? (nx * ny) : (sx * sy);
  double *local = (double*)malloc(local_size * sizeof(double));
  if (print_halo) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        #define IDX(i,j) ((j)*(nx) + (i))
        local[j*nx + i] = plane->data[IDX(i,j)];
        #undef IDX
      }
    }
  } else {
    for (int j = 1; j <= sy; j++) {
      for (int i = 1; i <= sx; i++) {
        #define IDX(i,j) ((j)*(sx+2) + (i))
        local[(j-1)*sx + (i-1)] = plane->data[IDX(i,j)];
        #undef IDX
      }
    }
  }

  double *recvbuf = NULL;
  if (Rank == 0) {
    recvbuf = (double*)malloc(Ntasks * local_size * sizeof(double));
  }
  MPI_Gather(local, local_size, MPI_DOUBLE, recvbuf, local_size, MPI_DOUBLE, 0, comm);
  free(local);

  if (Rank == 0) {
    if (print_halo) {
      if (iter>=0){
        printf("\n================ FULL GRID (%s, with halos) - Iteration %d ================\n", plane_label ? plane_label : "", iter);
      } else {
        printf("\n================ FULL GRID (%s, with halos) - FINAL STATE ================\n", plane_label ? plane_label : "");
      }
  for (int py = 0; py < (*Grid)[_y_]; py++) {
        for (int j = 0; j < ny; j++) {
          for (int px = 0; px < (*Grid)[_x_]; px++) {
            int proc = py * (*Grid)[_x_] + px;
            for (int i = 0; i < nx; i++) {
              int is_halo = (i == 0 || i == nx-1 || j == 0 || j == ny-1);
              if (is_halo)
                printf("[%6.2f]", recvbuf[proc*local_size + j*nx + i]);
              else
                printf(" %6.2f ", recvbuf[proc*local_size + j*nx + i]);
            }
            printf(" | ");
          }
          printf("\n");
        }
  for (int sep = 0; sep < (*Grid)[_x_]*nx; sep++) printf("--------");
        printf("\n");
      }
      printf("=======================================================\n\n");
    } else {
      if (iter>=0){
        printf("\n================ FULL GRID (%s, no halos) - Iteration %d ================\n", plane_label ? plane_label : "", iter);
      } else {
        printf("\n================ FULL GRID (%s, no halos) - FINAL STATE ================\n", plane_label ? plane_label : "");
      }
  for (int py = 0; py < (*Grid)[_y_]; py++) {
        for (int j = 0; j < sy; j++) {
          for (int px = 0; px < (*Grid)[_x_]; px++) {
            int proc = py * (*Grid)[_x_] + px;
            for (int i = 0; i < sx; i++) {
              printf("%8.3f ", recvbuf[proc*sx*sy + j*sx + i]);
            }
            printf(" | ");
          }
          printf("\n");
        }
  for (int sep = 0; sep < (*Grid)[_x_]*sx; sep++) printf("--------");
        printf("\n");
      }
      printf("====================================================\n\n");
    }
    free(recvbuf);
  }

  return 0;
}



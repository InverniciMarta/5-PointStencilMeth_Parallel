/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>

#include <sys/stat.h>


#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND_NORTH 0
#define RECV_NORTH 1
#define SEND_SOUTH 2
#define RECV_SOUTH 3
#define SEND_EAST 4
#define RECV_EAST 5
#define SEND_WEST 6
#define RECV_WEST 7

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

#define UPDATE_FULL_GRID 0
#define UPDATE_INNER_GRID 1


/* ==================================================================================================
=                                                                                                   =
=                                       CUSTOM DATA TYPES                                           =
=                                                                                                   =
===================================================================================================== */

typedef unsigned int uint;

typedef uint    vec2_t[2];
typedef double *restrict buffers_t[4];  // array of four pointers to double. restrict allows the compiler to optimize memory access: only one pointer will access the memory region at a time

typedef struct {
    double   * restrict data;
    vec2_t     size;
} plane_t;

// For timing
typedef struct {
    double init_time;
    double iter_time; 
    double inject_time;
    double setup_time;
    double wait_time;
    double inner_time;
    double border_time;
    double copy_time;
} timing_data_t;




/* ==================================================================================================
=                                                                                                   =
=                                       FUNCTION SIGNATURES                                         =
=                                                                                                   =
===================================================================================================== */
//.........................................
//functions defined in src
int initialize ( MPI_Comm *,
                int       ,
                int       ,
                int       ,
                char    **,
                vec2_t   *,
                vec2_t   *,                 
                int      *,
                int      *,
                int      *,
                uint      *,
                int      *,
                int      *,
                int      *,
                vec2_t  **,
                double   *,
                plane_t  *,
                buffers_t *,
                long int *, 
                int     *,
                int     *,
                int     *,
                int     *,
                int     *
                );

uint simple_factorization( uint, int *, uint ** );

int initialize_sources( int, int, MPI_Comm  *, uint [2], int, int *, vec2_t  **, long int *);

int memory_allocate (const int *, buffers_t *, plane_t *, int *);

int memory_release (plane_t *, buffers_t *);

int output_energy_stat ( int, plane_t * , double, int, MPI_Comm *);

int copy_buffers ( int *, MPI_Request *, buffers_t *, plane_t *);

int print_full_grid(MPI_Comm, int, int, vec2_t *, plane_t *, int, const char *, int);




//............................................
// functions defined in header
extern int inject_energy ( const int      ,
                           const int      ,
                           const vec2_t  *,
                           const double   ,
                                 plane_t *,
                           const vec2_t   
                        );


extern int update_plane ( const int       ,
                          const vec2_t   ,
                          const plane_t *,
                                plane_t * ,
                                int,
                                int *
                        );

extern int update_borders ( const int, const vec2_t, const plane_t *, plane_t * );

extern int get_total_energy( plane_t *, double  * );
            
int print_grid(plane_t *); 

int debug_print_all_grids(MPI_Comm , int , int , int , const char *, plane_t *);


                








/* ==================================================================================================
=                                                                                                   =
=                                       FUNCTION DEFINITIONS                                        =
=                                                                                                   =
===================================================================================================== */

inline int inject_energy ( const int      periodic,
                           const int      Nsources,
			               const vec2_t  *Sources,
			               const double   energy,
                           plane_t *plane,
                           const vec2_t   N
                           ) 
{
    const uint register sizex = plane->size[_x_]+2; //size is the correct size of the MPIprocess plane
    const uint register sizey = plane->size[_y_]+2; 
    double * restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*sizex + (i) )
    for (int s = 0; s < Nsources; s++)
        {
            //coordinates of the source in the process data grid, rememeber coordinates start from 1
            int x = Sources[s][_x_];
            int y = Sources[s][_y_];
            
            data[ IDX(x,y) ] += energy;
            

            //periodic has already determined the neighbours of the task, but we haven't defined the periodic for the case in which the MPI grid has width == 1 on either one or both directionswe (x,y)
            //for the non periodic case there's no problem since the outer layers for each process data grid is already initialized to 0 when the actual computational data grid is initialized to 0
            if ( periodic ) //With periodic boundaries conditions, if we inject energy at the border, we also inject it in the ghost cells
                { 
                    if ( (N[_x_] == 1)) //if the MPI grid is a vertical vector (north, south neighbours already defined)
                        {
                            // propagate the east-west boundaries of the process itself
                            // from the serial version
                            if ( x == 1 )
                                data[ IDX(sizex-1, y) ] += energy; //ghost cell on the left
                            if ( x == sizex-2)
                                data[ IDX(0, y) ] += energy; //ghost cell on the right
                        }
                    
                    if ( (N[_y_] == 1) ) //if the MPI grid is a horizontal vector (east and west neighbours already defined)
                        {
                            // propagate the north-south boundaries of the process itself
                            // from the serial version
                            if ( y == 1 )
                                data[ IDX(x, sizey-1) ] += energy; //ghost cell on the bottom
                            if ( y == sizey-2 )
                                data[ IDX(x, 0) ] += energy; //ghost cell on the top
                        }
                }                
        } 
 #undef IDX
    
  return 0;
}


//NOT ORIGINAL DEFINIITON OF UPDATE_PLANE. Added inner_grid boolean parameter, set to 0 by default, to choose whether to update the entire grid (original behaviour) or only the inner grid (1)
//Would be better to have a specialized function to compute inner grid updates? insteadof using a conditional inside a loop?
inline int update_plane(const int periodic, const vec2_t   N, const plane_t *oldplane, plane_t *newplane, int inner_grid, int *scan_grid_by_blocks)
{
    uint register fxsize = oldplane->size[_x_]+2; //padded size of the old plane, +2 for the outer layer
    uint register fysize = oldplane->size[_y_]+2;
    uint xsize = oldplane->size[_x_];
    uint ysize = oldplane->size[_y_];
    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    const double alpha = 0.5;
    const double betha = (1 - alpha) / 4.0;



    #define IDX(i,j) ((j)*fxsize + (i))

    if (!(*scan_grid_by_blocks)){
        #pragma omp parallel for schedule(static) //all variables defined ahead are implicitly shared
        for (uint j = 0; j < fysize; j++)
            for (uint i = 0; i < fxsize; i++)
            {
                if (!inner_grid || (i > 1 && i < xsize && j > 1 && j < ysize)) { //if inner_grid==0, update all the grid; if inner_grid==1, update only the inner grid (do not update the border cells).
                    double result = old[IDX(i,j)] * alpha;
                    double sum_i  = (old[IDX(i-1, j)] + old[IDX(i+1, j)]);
                    double sum_j  = (old[IDX(i, j-1)] + old[IDX(i, j+1)]);
                    result += (sum_i + sum_j)*betha;
                    new[IDX(i,j)] = result;
                }
                // If inner_grid==1 and data cells is on the border, do not write anything
            }  
    } else {    
    // Block size (hard coded here, could be better parameterized)
        const int block_size_x = 64;
        const int block_size_y = 48;

        # pragma omp parallel for collapse(2) schedule(static) //parallelize the outer two loops
        for (uint j = 0; j < fysize; j += block_size_y)
            for ( uint i = 0; i < fxsize; i += block_size_x)
                {
                    for (uint jj = j; jj < j + block_size_y && jj < fysize; jj++)
                        for ( uint ii = i; ii < i + block_size_x && ii < fxsize; ii++)
                            {
                                if ( !inner_grid || (ii > 1 && ii < xsize && jj > 1 && jj < ysize)) { //if inner_grid==0, update all the grid; if inner_grid==1, update only the inner grid (do not update the border cells).
                                double result = old[ IDX(ii,jj) ] * alpha;
                                double sum_i  = (old[IDX(ii-1, jj)] + old[IDX(ii+1, jj)]) ;
                                double sum_j  = (old[IDX(ii, jj-1)] + old[IDX(ii, jj+1)]);
                                result += (sum_i + sum_j )*betha;
                                new[ IDX(ii,jj) ] = result;
                                }
                            }
                } 
    }

    #undef IDX

    
    if ( !inner_grid && periodic  ){ //With periodic boundaries conditions, on thin side of MPI grid exchange borders within the same process.  x && y, true if both true, if x is false, y is not evaluated
            #define IDX( i, j ) ( (j)*fxsize + (i) )
            if ( (N[_x_] == 1)) //if the MPI grid is a vertical vector (north, south neighbours already defined)
                {
                    // propagate the east-west boundaries of the process itself
                    for ( uint j = 1; j <= ysize; j++)
                        {
                            new[ IDX(0, j) ] = new[ IDX(xsize, j) ]; //ghost cell on the right
                            new[ IDX(fxsize-1, j) ] = new[ IDX(1, j) ]; //ghost cell on the left
                        }
                }
            
            if ( (N[_y_] == 1) ) //if the MPI grid is a horizontal vector (east and west neighbours already defined)
                {
                    // propagate the north-south boundaries of the process itself
                    for ( uint i = 1; i <= xsize; i++)
                        {
                            new[ IDX(i, 0) ] = new[ IDX(i, ysize) ]; //ghost cell on the top
                            new[ IDX(i, fysize-1) ] = new[ IDX(i, 1) ]; //ghost cell on the bottom
                        }
                }
            #undef IDX
    }
    return 0;
}



inline int update_borders(const int periodic, const vec2_t   N, const plane_t *oldplane, plane_t *newplane )
{
    uint register fxsize = oldplane->size[_x_]+2; //padded size of the old plane, +2 for the outer layer
    uint register fysize = oldplane->size[_y_]+2;
    uint xsize = oldplane->size[_x_];
    uint ysize = oldplane->size[_y_];
    double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    const double alpha = 0.5;
    const double betha = (1 - alpha) / 4.0; 

    #define IDX(i,j) ((j)*fxsize + (i))

    #pragma omp parallel for schedule(static) //all variables defined ahead are implicitly shared
        for (uint j = 1; j <= ysize; j += ysize-1)
            for (uint i = 1; i <= xsize; i++)
            {
                    double result = old[IDX(i,j)] * alpha;
                    double sum_i  = (old[IDX(i-1, j)] + old[IDX(i+1, j)]);
                    double sum_j  = (old[IDX(i, j-1)] + old[IDX(i, j+1)]);
                    result += (sum_i + sum_j)*betha;
                    new[IDX(i,j)] = result;


            }
            // If inner_grid==1 and data cells is on the border, do not write anything 
        

    #pragma omp parallel for schedule(static)
        for (uint j = 2; j < ysize; j++)
            for (uint i = 1; i <= xsize; i += xsize-1)
            {
                    double result = old[IDX(i,j)] * alpha;
                    double sum_i  = (old[IDX(i-1, j)] + old[IDX(i+1, j)]);
                    double sum_j  = (old[IDX(i, j-1)] + old[IDX(i, j+1)]);
                    result += (sum_i + sum_j)*betha;
                    new[IDX(i,j)] = result;
            }
    

    #undef IDX

   if (periodic){ //With periodic boundaries conditions, on thin side of MPI grid exchange borders within the same process.  x && y, true if both true, if x is false, y is not evaluated
            #define IDX( i, j ) ( (j)*fxsize + (i) )
            if ( (N[_x_] == 1)) //if the MPI grid is a vertical vector (north, south neighbours already defined)
                {
                    // propagate the east-west boundaries of the process itself
                    for ( uint j = 1; j <= ysize; j++)
                        {
                            new[ IDX(0, j) ] = new[ IDX(xsize, j) ]; //ghost cell on the right
                            new[ IDX(fxsize-1, j) ] = new[ IDX(1, j) ]; //ghost cell on the left
                        }
                }
            
            if ( (N[_y_] == 1) ) //if the MPI grid is a horizontal vector (east and west neighbours already defined)
                {
                    // propagate the north-south boundaries of the process itself
                    for ( uint i = 1; i <= xsize; i++)
                        {
                            new[ IDX(i, 0) ] = new[ IDX(i, ysize) ]; //ghost cell on the top
                            new[ IDX(i, fysize-1) ] = new[ IDX(i, 1) ]; //ghost cell on the bottom
                        }
                }
            #undef IDX
        }
    return 0; 
}



inline int get_total_energy( plane_t *plane, double  *energy )
{

    const int register xsize = plane->size[_x_]; //the computational data grid size of the process without the outer layer
    const int register ysize = plane->size[_y_];
    const int register fsize = xsize+2;

    double * restrict data = plane->data; //the padded data array(data grid) of the process 
    
   #define IDX( i, j ) ( (j)*fsize + (i) )

   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif


    #pragma omp parallel for reduction(+: totenergy) schedule(static) //(static is the default)
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ]; 

    
   #undef IDX

    *energy = (double)totenergy;
    return 0;
}



static int file_exists_and_nonempty(const char *path) 
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (st.st_size > 0);
}


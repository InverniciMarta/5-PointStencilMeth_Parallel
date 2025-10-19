CC=mpicc
CFLAGS=-fopenmp -O3 -g -march=native -Wall -Wpedantic -Werror
LDFLAGS=-lm

all: stencil5

stencil5: stencil5.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f stencil5
	rm -f *.o
	rm -f *.csv


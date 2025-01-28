CC=mpicc
BINS=mt-p2p-msgrate-multicomm mt-p2p-msgrate-singlecomm
CFLAGS=-fopenmp

all: $(BINS)

mt-p2p-msgrate-multicomm: mt-p2p-msgrate.c
	$(CC) $(CFLAGS) -o mt-p2p-msgrate-multicomm mt-p2p-msgrate.c

mt-p2p-msgrate-singlecomm: mt-p2p-msgrate.c
	$(CC) $(CFLAGS) -DSINGLECOMM -o mt-p2p-msgrate-singlecomm mt-p2p-msgrate.c

clean:
	rm -f $(BINS)

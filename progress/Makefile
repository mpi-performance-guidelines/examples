CC=mpicc
BINS=async1 async1-s async2

all: $(BINS)

async1-s: async1.c
	$(CC) $(CFLAGS) -DSLEEP -o async1-s async1.c

clean:
	rm -f $(BINS)

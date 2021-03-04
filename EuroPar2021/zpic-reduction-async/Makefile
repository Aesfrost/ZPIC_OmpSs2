# PGI options
CC = pgimcc
CFLAGS = --ompss-2 -O3

# Intel Compiler
#CC = imcc
#CFLAGS = --ompss-2 -O3 -std=c99 -mtune=skylake --Wn,-xCORE-AVX512,-xHost

# GCC options
# CC = mcc
# CFLAGS = --ompss-2 -O3 -std=c99 -DTEST

INCLUDES = -I/home/nicolas/ompss-2/include/
LDFLAGS = -lm

SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c region.c 
TARGET = zpic

all : $(SOURCE) $(TARGET)

valgrind: $(SOURCE)
	gcc $^ $(INCLUDES) -o $(TARGET) -O3 -std=c99 -g $(LDFLAGS) 
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 2

$(TARGET) : $(SOURCE:.c=.o) $(KERNELS:.c=.o)
	$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

%.o : %.c
	$(CC) -c $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o

PREFIX=.

# GCC options
CC = /home/nicolas/ompss/bin/mcc
CFLAGS = --ompss -O3 --opencl

INCLUDES =
LDFLAGS = -lm

SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c csv_handler.c 
KERNELS = #kernel_gpu.cl
TARGET = zpic

all : $(SOURCE) $(TARGET)

valgrind: $(SOURCE)
	gcc $^ $(INCLUDES) -o $(TARGET) $(LDFLAGS) -lOpenCL -g -O3
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 2

$(TARGET) : $(SOURCE:.c=.o) $(KERNELS:.c=.o)
	$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

%.o : %.c
	$(CC) -c $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o mcc_*

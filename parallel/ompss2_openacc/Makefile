# GCC options
CC = pgimcc
CFLAGS = --ompss-2 -O3 

#CC = pgcc
#CFLAGS = -acc -ta=tesla:managed -Minfo=acc -O3 -fast -Mcuda=lineinfo -g

INCLUDES = -I/home/nicolas/ompss-2/include/
LDFLAGS = -lm

KERNELS = kernel_particles.c kernel_emf.c kernel_current.c
SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c csv_handler.c region.c 
TARGET = zpic

all : $(TARGET)

valgrind: $(SOURCE) $(KERNELS)
	gcc $^ $(INCLUDES) -o $(TARGET) -O3 -std=c99 -g $(LDFLAGS) 
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 2 0.5 1

$(TARGET) : $(KERNELS:.c=.o) $(SOURCE:.c=.o)
	$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

%.o : %.c
	$(CC) -c $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o *.i

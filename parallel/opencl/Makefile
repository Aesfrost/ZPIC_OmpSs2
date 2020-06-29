# GCC options
CC = $(HOME)/ompss/bin/mcc
CFLAGS = --ompss -O2 -std=c99 -Wall -g -k

#CC = pgcc
#CFLAGS = -acc -ta=tesla:managed -Minfo=acc -O3 -fast -Mcuda=lineinfo -g

INCLUDES = -I/home/nicolas/ompss-2/include/
LDFLAGS = -lm

KERNELS =
SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c region.c 
TARGET = zpic

all : $(TARGET)

valgrind: $(SOURCE) $(KERNELS)
	gcc $^ $(INCLUDES) -o $(TARGET) -O3 -std=c99 -g $(LDFLAGS) 
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 2 0.5 1

$(TARGET) : $(SOURCE) $(KERNELS)
	$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o mcc_*.c

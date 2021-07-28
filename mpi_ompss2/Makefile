# GCC options
CC = gcc
CFLAGS = -std=c99 -Wall -O3 -g -fopenmp -DTEST

INCLUDES = 
LDFLAGS = -lm

SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c region.c utilities.c task_management.c
TARGET = zpic

OMPSS2_HOME = /home/nicolas/ompss-2

all : $(TARGET)

tasking : CC = mcc
tasking : INCLUDES += -I$(OMPSS2_HOME)/include
tasking : LDFLAGS += -L$(OMPSS2_HOME)/lib -lnanos6 
tasking : CFLAGS += -DENABLE_TASKING --ompss-2 
tasking : $(TARGET)

valgrind: $(SOURCE)
	mpicc $^ $(CFLAGS) -o $(TARGET) $(INCLUDES) $(LDFLAGS)
	mpirun -np 4 valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 8

# $(TARGET) : $(SOURCE:.c=.o)
# 	mpiicc -cc=$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)
# 
# %.o : %.c
# 	mpiicc -cc=$(CC) -c $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

$(TARGET) : $(SOURCE:.c=.o)
	$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS) $(shell mpicc --showme:link)

%.o : %.c
	$(CC) -c $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS) $(shell mpicc --showme:compile)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o .gpi2.* mcc_*

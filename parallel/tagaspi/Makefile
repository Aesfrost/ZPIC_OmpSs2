# GCC options
CC = mcc
CFLAGS = -std=c99 -Wall -O3 -g --ompss-2 -K

GPI2_HOME = /home/nicolas/gaspi
TAGASPI_HOME = /home/nicolas/tagaspi
# GPI2_HOME = /home/bsc85/bsc85186/gaspi/install/
INCLUDES = -I$(GPI2_HOME)/include -I$(TAGASPI_HOME)/include

LDFLAGS = -lm
LDFLAGS += -L$(GPI2_HOME)/lib64 -L$(TAGASPI_HOME)/lib/ -ltagaspi -lnuma -lstdc++ -lGPI2-dbg -lpthread -lrt -libverbs

SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c region.c utilities.c
TARGET = zpic

all : $(TARGET)

valgrind: $(SOURCE)
	mpicc $^ $(CFLAGS) -o $(TARGET) $(INCLUDES) $(LDFLAGS)
	mpirun -np 1 valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 2

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
	rm -f $(TARGET) *.o .gpi2.* mcc_* machine_* *_r*_t1

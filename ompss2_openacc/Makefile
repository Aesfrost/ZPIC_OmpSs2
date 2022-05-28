# GCC options
CC = pgimcc
CFLAGS = --ompss-2 -O3 --openacc -DTEST --cuda -DENABLE_ADVISE

#CC = pgcc
#CFLAGS = -acc -ta=tesla:managed:cuda10.1 -Minfo=acc -O3 -fast -Mcuda=ptxinfo -g

INCLUDES = 
LDFLAGS = -lm -lcuda

SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c utilities.c region.c 
TARGET = zpic

all: $(TARGET)

affinity: CFLAGS += -DENABLE_AFFINITY --cuda
affinity: all

prefetch: CFLAGS += -DENABLE_PREFETCH
prefetch: affinity

valgrind: $(SOURCE)
	gcc $^ $(INCLUDES) -o $(TARGET) -O3 -std=c99 -g $(LDFLAGS) -Wall
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 4 0.5 2

$(TARGET) : $(SOURCE:.c=.o)
	$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

%.o : %.c
	$(CC) -c $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o *.i

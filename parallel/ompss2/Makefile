# PGI options
CC = pgimcc
CFLAGS = --ompss-2 -O3

# GCC options
# CC = mcc
# CFLAGS = --ompss-2 -O3 -std=c99

INCLUDES = -I/home/nicolas/ompss-2/include/
LDFLAGS = -lm

SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c csv_handler.c region.c 
TARGET = zpic

all : $(SOURCE) $(TARGET)

valgrind: $(SOURCE)
	gcc $^ $(INCLUDES) -o $(TARGET) -O3 -std=c99 -g $(LDFLAGS) 
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 2

$(TARGET) : $(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o

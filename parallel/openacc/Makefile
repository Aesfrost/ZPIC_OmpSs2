# GCC options
CC = pgcc
CFLAGS = -acc -ta=tesla:managed -Minfo=acc -fast -O3 -Mcuda=lineinfo
LDFLAGS = -lm

KERNELS = kernel_particles.c kernel_current.c kernel_emf.c
SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c csv_handler.c 
TARGET = zpic
OBJ = $(SOURCE:.c=.o) $(KERNELS:.c=.o)

all : $(SOURCE) $(TARGET)

$(TARGET) : $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

.c.o:
	$(CC) -c $(CFLAGS) $< $(LDFLAGS) -o $@

valgrind: $(SOURCE) $(KERNELS)
	gcc $^ $(INCLUDES) -o $(TARGET) -O3 -std=c99 -g $(LDFLAGS) 
	./zpic
	#valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET)
	#rm -r *.o

clean:
	@touch $(TARGET) $(OBJ)
	rm -f $(TARGET) $(OBJ)

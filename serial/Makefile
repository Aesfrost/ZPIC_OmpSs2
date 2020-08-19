# PGI options
# CC = pgcc
# CFLAGS = -fast -O3

# GCC compiler
CC = gcc
CFLAGS = -O3 -std=c99 -pedantic

# Clang options
#CC = clang
#CFLAGS = -Ofast

LDFLAGS = -lm

SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c csv_handler.c

TARGET = zpic

OBJ = $(SOURCE:.c=.o)

all : $(SOURCE) $(TARGET)

$(TARGET) : $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

.c.o:
	$(CC) -c $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	@touch $(TARGET) $(OBJ)
	rm -f $(TARGET) $(OBJ)

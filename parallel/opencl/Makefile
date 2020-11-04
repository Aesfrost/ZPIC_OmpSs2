# GCC options
CC = /home/nicolas/ompss/bin/mcc
CFLAGS = --ompss -O3 --opencl

INCLUDES = -I.
LDFLAGS = -lm -lOpenCL

SOURCE = current.c emf.c particles.c random.c timer.c main.c simulation.c zdf.c csv_handler.c 
FPGA_KERNEL = kernel_fpga
TARGET = zpic

all : gpu

cpu : $(TARGET)

gpu : CFLAGS += -DTARGET_GPU=1
gpu : $(TARGET)

fpga : CFLAGS += -DTARGET_FPGA=1
fpga : $(SOURCE:.c=.o) $(FPGA_KERNEL).aocx
	$(CC) $^ -o $(TARGET) $(CFLAGS) $(INCLUDES) $(LDFLAGS)

compile_fpga_obj:
	aoc $(FPGA_KERNEL) -ffp-reassoc -report -ffp-contract=fast -v 

valgrind: $(SOURCE) 
	gcc $^ $(INCLUDES) -o $(TARGET) $(LDFLAGS) -lOpenCL -g -O3
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET) 2

$(TARGET) : $(SOURCE:.c=.o) 
	$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

%.o : %.c
	$(CC) -c $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o mcc_*

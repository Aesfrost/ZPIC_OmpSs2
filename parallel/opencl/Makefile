# EPEEC Cluster
#CC = mcc
#CFLAGS = --ompss -O3 --opencl -DCL_TARGET_OPENCL_VERSION=120
#
#INCLUDES = -I. -I/opt/rocm-3.8.0/opencl/include
#LDFLAGS = -lm -L/opt/rocm-3.8.0/opencl/lib/ -lOpenCL


# GCC options
CC = /home/nicolas/ompss/bin/mcc
CFLAGS = --ompss -O3 --opencl -DCL_TARGET_OPENCL_VERSION=120

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
	aoc $(FPGA_KERNEL) -ffp-reassoc -report -ffp-contract=fast -v -profile=all -incremental -seed=7

valgrind: $(SOURCE) 
	gcc $^ $(INCLUDES) -o $(TARGET) $(LDFLAGS) -lOpenCL -g -O3
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=log.txt ./$(TARGET)

$(TARGET) : $(SOURCE:.c=.o) 
	$(CC) $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

%.o : %.c
	$(CC) -c $^ -o $@ $(CFLAGS) $(INCLUDES) $(LDFLAGS)

clean:
	@touch $(TARGET) 
	rm -f $(TARGET) *.o mcc_*

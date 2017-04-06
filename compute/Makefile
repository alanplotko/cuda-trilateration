CC = nvcc
FLAGS = -Wno-deprecated-gpu-targets -std=c++11 -g -G
FILENAME = assign2
PROGNAME = trilateration

all: $(FILENAME)

$(FILENAME):
	$(CC) -o $(PROGNAME) $(FLAGS) $(FILENAME).cu

clean:
	rm $(PROGNAME)

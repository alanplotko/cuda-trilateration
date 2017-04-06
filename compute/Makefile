CC = nvcc
FLAGS = -Wno-deprecated-gpu-targets
FILENAME = assign2
PROGNAME = trilateration

all: $(FILENAME)

$(FILENAME):
	$(CC) -o $(PROGNAME) $(FLAGS) $(FILENAME).cu

clean:
	rm $(PROGNAME)

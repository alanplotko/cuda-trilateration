CC = g++
CPPFLAGS = -std=c++11 -Wall -pedantic
FILENAME = gencoords
OBJS = gencoords.o
PROGNAME = gen

all: $(FILENAME)

$(FILENAME): $(OBJS)
	$(CC) -o $(PROGNAME) $(FLAGS) $(FILENAME).cpp

clean:
	rm $(PROGNAME) $(OBJS)


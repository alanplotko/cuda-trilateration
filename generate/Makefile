CC = g++
CFLAGS = -std=c++11 -Wall -pedantic
FILENAME = gencoords
OBJS = gencoords.o
PROGNAME = gen

all: $(FILENAME)

$(FILENAME): $(OBJS)
	$(CC) -o $(PROGNAME) $(CFLAGS) $(FILENAME).cpp

%.o: %.cpp
	$(CC) $(CFLAGS) -o $@ -c $^

clean:
	rm $(PROGNAME) $(OBJS)


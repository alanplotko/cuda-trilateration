#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <utility>

typedef std::pair<double, double> Point;

std::ostream& operator<<(std::ostream& stream, Point& p) {
    return stream << p.first << " " << p.second;
}

// Kernel definition
__global__ void FindPoint(const Point &a, const Point &b, const Point &c, double da, double db, double dc) {
    int i = threadIdx.x;
    float A = (-2 * a.first) + (2 * b.first),
          B = (-2 * a.second) + (2 * b.second),
          C = (da * da) - (db * db) - (a.first * a.first) + (b.first * b.first) - (a.second * a.second) + (b.second * b.second),
          D = (-2 * b.first) + (2 * c.first),
          E = (-2 * b.second) + (2 * c.second),
          F = (db * db) - (dc * dc) - (b.first * b.first) + (c.first * c.first) - (b.second * b.second) + (c.second * c.second);
}

// Set up guard points
void setGuards(std::ifstream &ifs, Point &a, Point &b, Point &c) {
    double x, y;
    ifs >> x >> y;
    a = std::make_pair(x, y);
    ifs >> x >> y;
    b = std::make_pair(x, y);
    ifs >> x >> y;
    c = std::make_pair(x, y);
}

int main(int argc, char* argv[]) {
    // Process flags
    std::string filename;
    int flag;
    opterr = 0;
    while ((flag = getopt(argc, argv, "hi:")) != -1) {
        switch(flag) {
            case 'i':
                filename = optarg;
                break;
            case 'h':
                std::cerr << "Usage: ./gen [-vho] <file-path>\n\n" <<
                             "Options:\n" <<
                             "-h\t\t Show usage string and exit\n" <<
                             "-i <file-path>\t Read input from provided file\n";
                exit(-1);
            case '?':
                if (optopt == 'i') {
                    std::cerr << "Option -" << (char)optopt << " requires an argument.\n";
                } else if (isprint(optopt)) {
                    std::cerr << "Unknown option `-" << (char)optopt << "'.\n";
                } else {
                    std::cerr << "Unknown option character `\\x" << (char)optopt << "'.\n";
                }
                exit(-1);
            default:
                exit(-1);
        }
    }

    // Ensure filename was passed
    if (filename.empty()) {
        std::cerr << "Error: input filename required\n";
        exit(-1);
    }

    // Open input file for reading
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ios::in);
    if (!ifs.is_open()) {
        std::cerr << "Error: failed to open " << filename << "\n";
        exit(-1);
    }

    // Use API to determine U and V values
    int nDevices;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "\n";
    } else {
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cerr << "Device Number: " << i << "\n";
            std::cerr << "  Device name: " << prop.name << "\n";
            std::cerr << "  Memory Clock Rate (KHz): " << prop.memoryClockRate <<  "\n";
            std::cerr << "  Memory Bus Width (bits): " << prop.memoryBusWidth <<  "\n";
            std::cerr << "  Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << "\n";
        }
    }

    // Set up guard points
    Point a, b, c, p;
    setGuards(ifs, a, b, c);

    // Read in distances and determine point p
    double da, db, dc;
    while (true) {
        ifs >> da >> db >> dc;

        // Kernel invocation with N threads
        FindPoint<<<1, 1>>>(a, b, c, da, db, dc);

        if(ifs.eof()) break;
    }

    ifs.close();
    return 0;
}

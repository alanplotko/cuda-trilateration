#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <utility>

bool verifyMode = false;
typedef std::pair<double, double> Point;

std::ostream& operator<<(std::ostream& stream, Point& p) {
    return stream << p.first << " " << p.second;
}

inline double dist(const Point &a, const Point &b) {
    double dx = (a.first - b.first);
    double dy = (a.second - b.second);
    return sqrt((dx * dx) + (dy * dy));
}

inline void move(Point &p) {
    p.first += static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 5.0));
    p.second += static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 5.0));
}

inline Point genPair() {
    return std::make_pair(
        static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 100.0)),
        static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / 100.0))
    );
}

int main(int argc, char* argv[]) {
    // Verify args and confirm filename
    std::string filename = "coordinates.out";
    int flag;
    opterr = 0;
    while ((flag = getopt(argc, argv, "vho:")) != -1) {
        switch(flag) {
            case 'v':
                verifyMode = true;
                break;
            case 'o':
                filename = optarg;
                break;
            case 'h':
                std::cerr << "Usage: ./gen [-vho] <file-path>\n\n" <<
                             "Options:\n" <<
                             "-v\t\t Enable verify mode (default: off)\n" <<
                             "-h\t\t Show usage string and exit\n" <<
                             "-o <file-path>\t Write output to provided file (default: coordinates.out)\n";
                exit(-1);
            case '?':
                if (optopt == 'o') {
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

    // Set up guard points and initial distances
    srand(time(NULL));
    Point a = genPair(), b = genPair(), c = genPair(), p = genPair();
    double da, db, dc;

    // Open file for writing coordinates
    std::ofstream ofs;
    ofs.open(filename, std::ofstream::out | std::ofstream::trunc);

    // Write out guard points
    ofs << a << "\n" << b << "\n" << c << "\n";

    // Calculate distances
    const int iterations = 12;
    if (verifyMode) {
        double avgX = 0.0, avgY = 0.0;
        for (int i = 1; i <= iterations; i++) {
            da = dist(a, p);
            db = dist(b, p);
            dc = dist(c, p);
            avgX += p.first;
            avgY += p.second;
            if (i % 4 == 0) {
                avgX /= 4.0;
                avgY /= 4.0;
                ofs << avgX << " " << avgY << "\n";
                avgX = avgY = 0.0;
            }
            move(p);
        }
    } else {
        for (int i = 0; i < iterations; i++) {
            da = dist(a, p);
            db = dist(b, p);
            dc = dist(c, p);
            ofs << da << " " << db << " " << dc << "\n";
            move(p);
        }

    }

    ofs.close();
    return 0;
}
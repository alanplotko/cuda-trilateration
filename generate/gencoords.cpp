#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <utility>

typedef std::pair<double, double> Point;

std::ostream& operator<<(std::ostream& stream, Point& p) {
    return stream << p.first << " " << p.second;
}

inline double dist(const Point &a, const Point &b) {
    double dx = (a.first - b.first);
    double dy = (a.second - b.second);
    return sqrt((dx * dx) + (dy * dy));
}

inline double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

inline void move(Point &p) {
    p.first += fRand(-1.0, 1.0);
    p.second += fRand(-1.0, 1.0);
}

inline Point genPair() {
    return std::make_pair(fRand(0.0, 100.0), fRand(0.0, 100.0));
}

int main(int argc, char* argv[]) {
    // Verify args and confirm filename
    std::string genFilename = "coordinates.out";
    std::string verifyFilename = "verification.out";
    int flag;
    int iterations = -1;
    opterr = 0;
    while ((flag = getopt(argc, argv, "hi:v:o:")) != -1) {
        switch(flag) {
            case 'v':
                verifyFilename = optarg;
                break;
            case 'i':
                iterations = pow(2.0, atoi(optarg));
                break;
            case 'o':
                genFilename = optarg;
                break;
            case 'h':
                std::cerr << "Usage: ./gen [-vho] <file-path>\n\n" <<
                             "Options:\n" <<
                             "-v <file-path>\t Write verification output to provided file (default: verifcation.out)\n" <<
                             "-i <iterations>\t Will write out 2^<iterations> cases to try\n" <<
                             "-h\t\t Show usage string and exit\n" <<
                             "-o <file-path>\t Write output to provided file (default: coordinates.out)\n";
                exit(-1);
            case '?':
                if (optopt == 'o' || optopt == 'v' || optopt == 'i') {
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
    srand(0);//time(NULL));
    Point a = genPair(), b = genPair(), c = genPair(), p = genPair();
    double da, db, dc;

    // Open file for writing coordinates
    std::ofstream gen, verify;
    if (iterations == -1) {
        std::cerr << "Error: iteration count required\n";
        exit(-1);
    }

    std::cerr << "-------------------------\n";
    std::cerr << "Writing " << iterations << " cases\n";
    std::cerr << "-------------------------\n";

    gen.open(genFilename, std::ofstream::out | std::ofstream::trunc);
    verify.open(verifyFilename, std::ofstream::out | std::ofstream::trunc);

    // Write out guard points
    gen << a << "\n" << b << "\n" << c << "\n";

    // Calculate distances
    double avgX = 0.0, avgY = 0.0;
    for (int i = 1; i <= iterations; i++) {
        da = dist(a, p);
        db = dist(b, p);
        dc = dist(c, p);
        avgX += p.first;
        avgY += p.second;
        gen << da << " " << db << " " << dc << "\n";
        if (i % 4 == 0) {
            avgX /= 4.0;
            avgY /= 4.0;
            verify << avgX << " " << avgY << "\n";
            avgX = avgY = 0.0;
        }
        move(p);
    }

    gen.close();
    verify.close();
    return 0;
}

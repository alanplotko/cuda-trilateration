#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#define BLOCK_N 2 // 16 on Alan's PC
#define THREAD_N 96 // 2048 on Alan's PC

struct Point {
    double x;
    double y;
};

struct Dist {
    double da;
    double db;
    double dc;
    Dist(double a, double b, double c) : da(a), db(b), dc(c) {}
};

std::ostream& operator<<(std::ostream& stream, Point& p) {
    return stream << p.x << " " << p.y;
}

__device__ Point& operator+=(Point &a, const Point &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__device__ Point& operator/(Point &a, double d) {
    a.x /= d;
    a.y /= d;
    return a;
}

__device__ Point Trilaterate(const Point &a, const Point &b, const Point &c, const Dist &ref) {
    double A = (-2 * a.x) + (2 * b.x),
           B = (-2 * a.y) + (2 * b.y),
           C = (ref.da * ref.da) - (ref.db * ref.db) - (a.x * a.x) + (b.x * b.x) - (a.y * a.y) + (b.y * b.y),
           D = (-2 * b.x) + (2 * c.x),
           E = (-2 * b.y) + (2 * c.y),
           F = (ref.db * ref.db) - (ref.dc * ref.dc) - (b.x * b.x) + (c.x * c.x) - (b.y * b.y) + (c.y * c.y);
    if (A * E == D * B || B * D == E * A) { // Don't divide by 0
        return Point { 0, 0 };
    }
    return Point { ((C * E) - (F * B)) / ((A * E) - (D * B)), ((C * D) - (F * A)) / ((B * D) - (E * A)) };
}

// Kernel definition
__global__ void FindPoint(const Point a, const Point b, const Point c, const Dist *dst, size_t num, Point *pts) {
    __shared__ Point t[THREAD_N];
    // const int i = threadIdx.x;
    // const int vec_n = blockIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i + vec_n * THREAD_N < num) {
    if (i < num) {
        printf("BlockIdx=%d, ThreadIdx=%d, i=%d, dst[i]=(%f,%f,%f)\n",
                blockIdx.x, threadIdx.x, i, dst[i].da, dst[i].db, dst[i].dc);
        int tid = threadIdx.x;
        t[tid] = Trilaterate(a, b, c, dst[i]);//dst[i + vec_n * THREAD_N]);
        // Have each thread calculate the trilateration of a dst[i]
        __syncthreads();
        // Avg every 4 points, store in pts
        for (int fold = 2; fold > 0; fold /= 2) {
            if (tid % 4 < fold) { // 4-point segment
                t[tid] += t[tid + fold];
            }
            __syncthreads();
        }
        if (tid % 4 == 0) pts[i / 4] = t[tid] / 4;
    }
}

// Set up guard points
void setGuards(std::ifstream &ifs, Point &a, Point &b, Point &c) {
    ifs >> a.x >> a.y
        >> b.x >> b.y
        >> c.x >> c.y;
}

/* The current CUDA API goes up to 3.0 SM support (Kepler). For newer
 * architectures, like Maxwell and Pascal, we use this function that
 * many industry professionals have agreed on to determine the number of
 * cores. Until the API is updated to include the numbers for >3.0 SM, we
 * will use this function to get the count.
 */
int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else std::cerr << "Unknown device type\n";
            break;
        default:
            std::cerr << "Unknown device type\n";
            break;
    }
    return cores;
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
                std::cerr << "Usage: ./gen [-ho] <file-path>\n\n" <<
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
    int nDevices, U = -1, V = -1;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "\n";
        exit(-1);
    } else {
        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            U = prop.multiProcessorCount;
            V = getSPcores(prop);
        }
    }

    // Set up guard points
    Point a, b, c;
    setGuards(ifs, a, b, c);

    // Read in distances and determine point p
    if (U == -1 || V == -1) {
        std::cerr << "Error: Could not fetch information on number of multiprocessors or cores\n";
        exit(-1);
    }

    std::vector<Dist> data;

    while (true) {
        double da, db, dc;
        ifs >> da >> db >> dc;
        if (ifs.eof()) break;
        data.push_back(Dist(da, db, dc));
        //std::cerr << "Dist { " << da << " " << db << " " << dc << " }\n";
    }
    //std::cerr << "Size of dists: " << data.size() << "\n";
    ifs.close();

    Dist *dst;
    Point *pts;
    Point *res = new Point[data.size() / 4];
    err = cudaMalloc((void **)&dst, data.size() * sizeof(Dist));
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "\n";
        exit(-1);
    }
    err = cudaMalloc((void **)&pts, (data.size() / 4) * sizeof(Point));
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "\n";
        exit(-1);
    }

    err = cudaMemcpy(dst, data.data(), data.size() * sizeof(Dist), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "\n";
        exit(-1);
    }

    FindPoint<<<BLOCK_N, THREAD_N>>>(a, b, c, dst, data.size(), pts);

    err = cudaMemcpy(res, pts, (data.size() / 4) * sizeof(Point), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "\n";
        exit(-1);
    }

    for (int i = 0; i < data.size() / 4; i++) {
        std::cerr << res[i] << "\n";
    }

    cudaFree(dst);
    cudaFree(pts);
    delete[] res;

    return 0;
}

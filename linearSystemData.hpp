#ifndef LINEAR_SYSTEM_DATA
#define LINEAR_SYSTEM_DATA

#include <fstream>
#include <vector>
#include <iostream>

namespace WellSolver
{

    class LinearSystemData
    {
        public:
            const int block_m_;
            const int block_n_;
            std::vector<double> Dvals;
            std::vector<int> Dcols;
            std::vector<int> Drows;
            std::vector<double> Bvals;
            std::vector<int> Bcols;
            std::vector<int> Brows;
            std::vector<double> Cvals;
            std::vector<double> x;
            std::vector<double> y;

            LinearSystemData(const int block_m, const int block_n);

            template <class Scalar>
            void readVector(std::vector<Scalar>& vec, std::string filename) {
                std::ifstream input_file(filename);
                if (input_file.is_open()) {
                    Scalar num;
                    while (input_file >> num) {
                        vec.push_back(num);
                    }
                    input_file.close();
                } else {
                    std::cerr << "Error opening file." << std::endl;
                }
            }

            template <class Scalar>
            void saveVector(std::vector<Scalar> vec, std::string filename) {
                std::ofstream output_file(filename);
                if (output_file.is_open()) {
                    for (const auto& num : vec) {
                        output_file << num << " ";
                    }
                    output_file.close();
                } else {
                    std::cerr << "Error opening file." << std::endl;
                }
            }

            template <typename T>
            void printVector(const std::vector<T>& vec, const std::string& name) {
                std::cout << name <<": " << vec.size() << std::endl;
                std::cout << "[ ";
                for (const auto& val : vec) std::cout << val << " ";
                std::cout << "]";
                std::cout << std::endl;
                std::cout << std::endl;
            }
    };

}
#endif

#include <iostream>
#include <string>
#include <cstdlib>
#include <omp.h>
#include "InputFile.h"
#include "Driver.h"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: deqn <filename>" << std::endl;
        exit(1);
    }
   double  begin_time = omp_get_wtime();
    const char* filename = argv[1];
    InputFile input(filename);

    std::string problem_name(filename);

    int len = problem_name.length();

    if(problem_name.substr(len - 3, 3) == ".in")
        problem_name = problem_name.substr(0, len-3);

    // Strip out leading path
    size_t last_sep = problem_name.find_last_of("/");

    if (last_sep != std::string::npos) {
        last_sep = last_sep + 1;
    } else {
        last_sep = 0;
    }

    problem_name = problem_name.substr(last_sep, problem_name.size());

    Driver driver(&input, problem_name);

    driver.run();
   double finish_time =omp_get_wtime();

        std::cout<<"Overall Running time:"<<(finish_time-begin_time)*1000<<std::endl;
    return 0;
}

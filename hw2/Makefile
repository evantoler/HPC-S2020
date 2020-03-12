build:
	@g++ -std=c++11 -O3 val_test01_solved.cpp -o val_test01_solved
	@g++ -std=c++11 -O3 val_test02_solved.cpp -o val_test02_solved
	@g++ -std=c++11 -O3 -fopenmp -march=native MMult1.cpp -o MMult1
	@g++ -std=c++11 -O3 -fopenmp omp_solved2.c -o omp_solved2
	@g++ -std=c++11 -O3 -fopenmp omp_solved3.c -o omp_solved3
	@g++ -std=c++11 -O3 -fopenmp omp_solved4.c -o omp_solved4
	@g++ -std=c++11 -O3 -fopenmp omp_solved5.c -o omp_solved5
	@g++ -std=c++11 -O3 -fopenmp omp_solved6.c -o omp_solved6
	@g++ -std=c++11 -O3 -fopenmp jacobi2d-omp.cpp -o jacobi2d-omp
	@g++ -std=c++11 -O3 -fopenmp gs2d-omp.cpp -o gs2d-omp
	@echo "Compiled Successfully!"

clean:
	@rm -rf val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2d-omp gs2d-omp
	@echo "Cleaned Successfully!"

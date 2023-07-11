/*
This is a parallel program for two-phase fluid simulation based on explicit Navier-Stokes & Cahn-Hilliard system
Save it and Run with Visual Studio 2019（or later version） in Windows system
*/

#define _USE_MATH_DEFINES // for M_PI
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include "omp.h"
#include <chrono>
#include "helpers.h"
#include <filesystem>

using namespace std;
using namespace std::chrono;


int main(int argc, char** argv)
{
	//****************************************************Define Variables************************************************** 
		// MeshGrids	
	double lx = 1; // Length of domain in X
	double ly = 2; // Length of domain in Y
	int n_i = 64; //number of grid points in horizontal direction
	int n_j = 128;// number of grid points in vertical direction

	// Time & Delta time
	double total_time = 1.0;
	double dt = 1e-4; // time step length

	// constants for the equations
	double rho[2] = { 1,1000 }; // density rho, first value for phi = 1, second for phi = -1
	double eta[2] = { 0.1,10 }; // viscosity eta, first value for phi = 1, second for phi = -1
	double sigma = 24.9; // surface tension sigma
	double f[2] = { 0,-0.98 }; // force

	// Default Variables
	double m = 0.01; // mobility constant m

	double r = 0.25; // radius of the bubble
	double circle_center_x = 0.5;
	double circle_center_y = 0.5;

	double coord_x; // x coordinate of nodes
	double coord_y; // y coordinate of nodes
	double dam_width = 0.2;
	double dam_height = 0.2;

	// Variables Calculated
	double h = lx / n_i; // distance between grid points ("hx=hy" is necessary)
	double num_iterations = total_time / dt; // number of iterations
	double eps = 2.5 * h; // interface thickness epsilon 

	// constants which are relevant for simulation or runtime
	int print_every_round = num_iterations / 1000; // output every x rounds
	int cnt_threads = 8; // number of threads
	bool output_console = 1; // output in console (true = yes)
	bool output_file = 1; // output in files (true = yes)

	// output on screen
	//	cout<<"h="<<h<<"\n";
	cout << "lx                = " << lx << "\n";
	cout << "ly                = " << ly << "\n";
	cout << "n_i               = " << n_i << "\n";
	cout << "n_j               = " << n_j << "\n";
	cout << "total_time        = " << total_time << "\n";
	cout << "dt                = " << dt << "\n";
	cout << "rho[2]            = {" << rho[0] << ", " << rho[1] << "}\n";
	cout << "eta[2]            = {" << eta[0] << ", " << eta[1] << "}\n";
	cout << "sigma             = " << sigma << "\n";
	cout << "f[2]              = {" << f[0] << ", " << f[1] << "}\n";
	cout << "cnt_threads       = " << cnt_threads << "\n";
	cout << "output_console    = " << output_console << " (1 = True, 0 = False)\n";
	cout << "output_file       = " << output_file << " (1 = True, 0 = False)\n";
	cout << "\n";

	//Creat file "paraview" for save result.vtk
	string dir_path = "paraview";
	if (std::filesystem::exists(dir_path)) { // 如果文件夹已经存在
		std::filesystem::remove_all(dir_path); // 删除文件夹及其内容
	}
	create_directory(dir_path); // 创建文件夹

	// Buffer variables
	int cnt = 0;
	string progress_bar = "";
	double percent;

	// benchmark quantities
	double cnt_y_n = 0;
	double cnt_dvy_vy = 0;
	double cnt_dm_y = 0;

	// set sigma to correct value
	sigma = sigma * 3 / 2 / sqrt(2);

	// declaration of all data arrays
	Array2D phi_old(n_i + 2, n_j + 2); // phase field
	Array2D phi_new(n_i + 2, n_j + 2); // phase field
	Array2D mu(n_i + 2, n_j + 2); // mu
	Array2D p_new(n_i + 2, n_j + 2); // pressure
	Array2D p_old(n_i + 2, n_j + 2); // pressure
	Array2D rho_array(n_i + 2, n_j + 2); // density
	Array2D eta_array(n_i + 2, n_j + 2); // viscosity
	Array2D alpha(n_i + 2, n_j + 2);
	Array2D beta(n_i + 2, n_j + 2);
	Array2D gamma(n_i + 3, n_j + 3);
	Array2D u0(n_i + 3, n_j + 2); // velocity in horizontal direction
	Array2D u0_tilde(n_i + 3, n_j + 2); // u0 tilde
	Array2D u1(n_i + 2, n_j + 3); // velocity in vertical direction
	Array2D u1_tilde(n_i + 2, n_j + 3); // u1 tilde

	// prepare vtk output
	VTKWriter writer_phi("phi_", "paraview/");

	omp_set_num_threads(cnt_threads); // number of threads for OpenMP
	high_resolution_clock::time_point t1 = high_resolution_clock::now(); // time measurement

	if (output_console) {
		printf("\r[*] Last print to file no.: - | ETA: infs");
		fflush(stdout);
	}

	// set init condition - 2D droplet cases
#pragma omp parallel for schedule(static)
	for (int i = 0; i < n_i + 2; i++) {
		for (int j = 0; j < n_j + 2; j++) {
			phi_old(i, j) = tanh((r - sqrt(((i - 0.5) * h - circle_center_x) * ((i - 0.5) * h - circle_center_x) + ((j - 0.5) * h - circle_center_y) * ((j - 0.5) * h - circle_center_y))) / (sqrt(2.0) * eps));
			phi_new(i, j) = -1;
		}
	}



	if (output_file) {
		writer_phi.write_step(phi_old, cnt, 0, n_i + 1, 0, n_j + 1, h, -h / 2, -h / 2);
		writer_phi.finalize();
	}


	// calculation for num_iterations time steps
	for (int t = 0; t < num_iterations; t++) {
		// reset benchmark quantities every round
		cnt_y_n = 0;
		cnt_dvy_vy = 0;
		cnt_dm_y = 0;

#pragma omp parallel 
		{

			// Navier-Stokes equations
			// calculation of viscosity and density
#pragma omp for schedule(static)
			for (int i = 0; i < n_i + 2; i++) {
				for (int j = 0; j < n_j + 2; j++) {
					double phi_temp = phi_old(i, j);
					if (phi_temp > 1) phi_temp = 1;
					else if (phi_temp < -1) phi_temp = -1;

					eta_array(i, j) = ((phi_temp + 1) / 2) * (eta[0] - eta[1]) + eta[1];
					rho_array(i, j) = ((phi_temp + 1) / 2) * (rho[0] - rho[1]) + rho[1];
				}
			}

			// helper array gamma
#pragma omp for schedule(static)
			for (int i = 2; i < n_i + 1; i++) { // only from i=2 to n_i+1 due to free-slip condition at lateral boundaries
				for (int j = 1; j < n_j + 2; j++) {
					gamma(i, j) = 1 / (4 * h) * (u0(i, j) - u0(i, j - 1) + u1(i, j) - u1(i - 1, j))
						* (eta_array(i, j) + eta_array(i - 1, j) + eta_array(i, j - 1) + eta_array(i - 1, j - 1));
				}
			}

			// helper array alpha and beta
#pragma omp for schedule(static)
			for (int i = 0; i < n_i + 2; i++) {
				for (int j = 0; j < n_j + 2; j++) {
					alpha(i, j) = 2 * eta_array(i, j) * (u0(i + 1, j) - u0(i, j)) / h;
					beta(i, j) = 2 * eta_array(i, j) * (u1(i, j + 1) - u1(i, j)) / h;
				}
			}

			// calculation of temporary velocity
			// u0_tilde
#pragma omp for schedule(static)
			for (int i = 2; i < n_i + 1; i++) { // only from i=2 to n_i+1 due to vx=0 at lateral boundaries
				for (int j = 1; j < n_j + 1; j++) {
					u0_tilde(i, j) = ((
						(alpha(i, j) - alpha(i - 1, j) + gamma(i, j + 1) - gamma(i, j)) / h // viscosity
						+ ((mu(i, j) + mu(i - 1, j)) * (phi_old(i, j) - phi_old(i - 1, j)) * sigma) / (2 * h * eps) // surface tension
						)
						* 2 / (rho_array(i, j) + rho_array(i - 1, j)) // density
						- (u0(i, j) * (u0(i + 1, j) - u0(i - 1, j)) + 0.25 * (u1(i, j) + u1(i - 1, j)
							+ u1(i - 1, j + 1) + u1(i, j + 1)) * (u0(i, j + 1) - u0(i, j - 1))) / (2 * h) // convection
						)
						* dt + u0(i, j);
				}
			}

			// u1_tilde
#pragma omp for schedule(static)
			for (int i = 1; i < n_i + 1; i++) {
				for (int j = 2; j < n_j + 1; j++) {  // only from j=2 to n_j+1 due to vy=0 at lateral boundaries
					u1_tilde(i, j) = ((
						(beta(i, j) - beta(i, j - 1) + gamma(i + 1, j) - gamma(i, j)) / h // viscosity
						+ f[1] / 2 * (rho_array(i, j) + rho_array(i, j - 1) - 2.0 * rho[1]) // force
						+ ((mu(i, j) + mu(i, j - 1)) * (phi_old(i, j) - phi_old(i, j - 1)) * sigma) / (2 * h * eps) // surface tension
						)
						* 2 / (rho_array(i, j) + rho_array(i, j - 1)) // density
						- (0.25 * (u0(i, j) + u0(i, j - 1) + u0(i + 1, j) + u0(i + 1, j - 1)) * (u1(i + 1, j) - u1(i - 1, j))
							+ u1(i, j) * (u1(i, j + 1) - u1(i, j - 1))) / (2 * h) // convection
						)
						* dt + u1(i, j);
				}
			}

			// calculation of pressure p
			int iter = 10;
		for (int count = 0; count < iter; count++) { // solve iter time steps of pressure calculation 

#pragma omp for schedule(static)
				for (int i = 1; i < n_i + 1; i++) {
					for (int j = 1; j < n_j + 1; j++) {
						p_new(i, j) = (
							0.125 * (p_old(i + 1, j) + p_old(i - 1, j) + p_old(i, j + 1) + p_old(i, j - 1) - 4 * p_old(i, j))
							+ rho_array(i, j) * 0.125 / 4 * ((1 / rho_array(i + 1, j) - 1 / rho_array(i - 1, j)) * (p_old(i + 1, j) - p_old(i - 1, j))
								+ (1 / rho_array(i, j + 1) - 1 / rho_array(i, j - 1)) * (p_old(i, j + 1) - p_old(i, j - 1)))
							- h * rho_array(i, j) * 0.125 / dt * (u0_tilde(i + 1, j) - u0_tilde(i, j) + u1_tilde(i, j + 1) - u1_tilde(i, j))
							)
							+ p_old(i, j);
					}
				}

#pragma omp single
				{
					swap(p_old, p_new);
				}
			}
#pragma omp barrier
#pragma omp single 
			{
				swap(p_old, p_new); // finally swap p_old and p_new again to ensure p_new is the last computed value
			}

			// calculation of new velocities
			// u0
#pragma omp for schedule(static)
			for (int i = 2; i < n_i + 1; i++) { // only from i=2 to n_i+1 due to vx=0 at lateral boundaries
				for (int j = 1; j < n_j + 1; j++) {
					u0(i, j) = (-(p_new(i, j) - p_new(i - 1, j)) / h
						* 2 / (rho_array(i, j) + rho_array(i - 1, j)))
						* dt + u0_tilde(i, j);
				}
			}

			// u1
#pragma omp for schedule(static)
			for (int i = 1; i < n_i + 1; i++) { // only from j=2 to n_j+1 due to vy=0 at lateral boundaries
				for (int j = 2; j < n_j + 1; j++) {
					u1(i, j) = (-(p_new(i, j) - p_new(i, j - 1)) / h
						* 2 / (rho_array(i, j) + rho_array(i, j - 1)))
						* dt + u1_tilde(i, j);
				}
			}

			// Cahn-Hilliard equation
			// calculation of mu
#pragma omp for schedule(static)
			for (int i = 1; i < n_i + 1; i++) {
				for (int j = 1; j < n_j + 1; j++) {
					mu(i, j) = (-(eps * eps) / (h * h))
						* (phi_old(i + 1, j) + phi_old(i - 1, j) + phi_old(i, j + 1) + phi_old(i, j - 1) - 4 * phi_old(i, j))
						+ (phi_old(i, j) * phi_old(i, j) * phi_old(i, j)) - phi_old(i, j);
				}
			}

			// calculation of phase field phi
#pragma omp for schedule(static)
			for (int i = 1; i < n_i + 1; i++) {
				for (int j = 1; j < n_j + 1; j++) {
					phi_new(i, j) = ((m / (h * h))
						* (mu(i + 1, j) + mu(i - 1, j) + mu(i, j + 1) + mu(i, j - 1) - 4 * mu(i, j))
						- (((u0(i, j) + u0(i + 1, j)) / 2) * (phi_old(i + 1, j) - phi_old(i - 1, j))
							+ ((u1(i, j) + u1(i, j + 1)) / 2) * (phi_old(i, j + 1) - phi_old(i, j - 1))) / (2 * h))
						* dt + phi_old(i, j);
				}
			}

			// reset boundaries to correct boundary conditions
#pragma omp for schedule(static)
			for (int i = 0; i < n_i + 2; i++) {
				p_new(i, 0) = p_new(i, 1);
				p_new(i, n_j + 1) = p_new(i, n_j);
				phi_new(i, 0) = phi_new(i, 1);
				phi_new(i, n_j + 1) = phi_new(i, n_j);
				u0(i, 0) = -u0(i, 1);
				u0(i, n_j + 1) = -u0(i, n_j);
			}
#pragma omp for schedule(static)
			for (int j = 0; j < n_j + 2; j++) {
				p_new(0, j) = p_new(1, j);
				p_new(n_i + 1, j) = p_new(n_i, j);
				phi_new(0, j) = phi_new(1, j);
				phi_new(n_i + 1, j) = phi_new(n_i, j);
				u1(0, j) = -u1(1, j);
				u1(n_i + 1, j) = -u1(n_i, j);
			}
		}

		// swap fields of pressure and phase
		swap(p_old, p_new);
		swap(phi_new, phi_old);

		//  broken time ******
		double sum_residual = 0;
#pragma omp for schedule(static)
		for (int i = 1; i < n_i + 1; i++) {
			for (int j = 1; j < n_j + 1; j++) {
				sum_residual = (phi_new(i, j) - phi_old(i, j)) / (n_i * n_j) + sum_residual;
			}
		}

		if (abs(sum_residual) > 100 || isnan(sum_residual)) {
			cout << "\n\n" << "sum_residual=" << sum_residual << endl;
			cout << "num_loop=" << t << "\nsimulation is failed" << endl;
			break;
		}

		// output
		if (output_file)
			if (t % print_every_round == 0 || t == num_iterations - 1) {
				cnt++;
				writer_phi.write_step(phi_old, cnt, 0, n_i + 1, 0, n_j + 1, h, -h / 2, -h / 2);
				writer_phi.finalize();

				if (output_console) {
					// print progress
					high_resolution_clock::time_point t2 = high_resolution_clock::now();
					duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
					double seconds = ((time_span.count() / cnt) * (num_iterations / print_every_round) - time_span.count());
					double mins = ((time_span.count() / cnt) * (num_iterations / print_every_round) - time_span.count()) / 60;
					printf("\r[*] Last print to file no.: %i | ETA: %5.fs OR %3.fmin", cnt, seconds, mins);
					fflush(stdout);
				}
			}
	}

	if (output_console) {
		high_resolution_clock::time_point t3 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t3 - t1);
		printf("\nTime needed: %f\n", time_span.count());
	}


	return 0;
}

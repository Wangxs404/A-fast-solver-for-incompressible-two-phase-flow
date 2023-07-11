/*
* Array2D and VTKWriter class
* Author: Lucas D. Wittwer (https://github.com/lucaswittwer)
*/
#include <vector>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;
void create_directory(const std::string& dir_path) {
	fs::create_directory(dir_path);
}

// 2D array for storing a field variable
// in row-order => inner-most loop over j, not i!
// Example:
// 
//		Array2D u(N+1, N+1);
//		u(i, j) = 10.0;
class Array2D {

private:
	std::vector<double> data_;
	int isize_, jsize_;

public:
		Array2D(int isize, int jsize) : data_(isize * jsize, 0.0), isize_(isize), jsize_(jsize) { //data_`的初始容量为`isize * jsize`，每个元素被初始化为`0.0`
			assert(isize >= 0 && jsize >= 0);   //断言语句，用于在程序运行时进行条件检查，如果条件不满足，则会中止程序执行并输出错误信息。
		}

		//__attribute__((always_inline)) // gcc内联语句
		__forceinline //windows 内联语句
		inline const double& operator()(int i, int j) const { //定义函数用于访问矩阵元素，类似于用一维向量存储并访问
			assert(i >= 0 && i < isize_ && j >= 0 && j < jsize_);
			return data_[i * jsize_ + j];
		}


		__forceinline
		inline double& operator()(int i, int j) {
			assert(i >= 0 && i < isize_ && j >= 0 && j < jsize_);
			return data_[i * jsize_ + j];
		}


};

// Writes a StructuredGrid files in the VTK format and summarise them in as a PVD file for ParaView
class VTKWriter {

public:
	VTKWriter(std::string file_path_prefix, std::string folder) : file_path_prefix_(file_path_prefix), steps_(), folder_(folder) {}

	bool write_step(Array2D u, int step_number, int isize0, int isize, int jsize0, int jsize, double h=1, double offset_x=0, double offset_y=0) {

		std::string file_path = folder_ + file_path_prefix_ + std::to_string(step_number) + std::string(".vts");

		FILE * out = fopen(file_path.c_str(), "w");

		if (out == NULL) {
			return false;
		}

		fprintf(out, "<VTKFile type = \"StructuredGrid\" version = \"1.0\" byte_order = \"LittleEndian\" header_type = \"UInt64\">\n");
		fprintf(out, "  <StructuredGrid WholeExtent=\"%d %d %d %d 0 0\">\n", jsize0, jsize, isize0, isize);
		fprintf(out, "  <Piece Extent=\"%d %d %d %d 0 0\">\n",  jsize0, jsize, isize0, isize);
		fprintf(out, "    <Points>\n");
		fprintf(out, "      <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
		for (int i = isize0; i <= isize; i++) {
		       for (int j = jsize0; j <= jsize; j++) {
		              fprintf(out, "%f %f %f ", h * i + offset_x, h * j + offset_y, 0.0);
		       }
		}
		fprintf(out, "\n      </DataArray>\n");
		fprintf(out, "    </Points>\n");
		fprintf(out, "    <PointData Scalars=\"u\">\n");
		fprintf(out, "      <DataArray type=\"Float32\" Name=\"u\" format=\"ascii\">\n");
		for (int i = isize0; i <= isize; i++) {
		    for (int j = jsize0; j <= jsize; j++) {
		    	fprintf(out, "%f ", u(i,j));
		    }
		}
		fprintf(out, "\n      </DataArray>\n");
		fprintf(out, "    </PointData>\n");
		fprintf(out, "    <CellData> \n </CellData>\n");
		fprintf(out, "  </Piece>\n");
		fprintf(out, "</StructuredGrid>\n");
		fprintf(out, "</VTKFile>\n");


		fclose(out);

		steps_.push_back(step_number);
		return true;
	}

	bool finalize() {

		std::string pvd_file_name = folder_ + file_path_prefix_ + std::string(".pvd");

		FILE * out = fopen(pvd_file_name.c_str(), "w");

		if (out == NULL) {
			return false;
		}

        fprintf(out, "<?xml version=\"1.0\"?>\n");
        fprintf(out, "<VTKFile type=\"Collection\" version=\"0.1\">\n");
        fprintf(out, "  <Collection>\n");

        for (auto step : steps_) {

        	std::string file = file_path_prefix_ + std::to_string(step) + std::string(".vts");

			fprintf(out, "    <DataSet timestep=\"%d\" group=\"\" part=\"0\"\n", step);
			fprintf(out, "             file=\"%s\"/>\n", file.c_str());
		}
        fprintf(out, "  </Collection>\n");
        fprintf(out, "</VTKFile>");

        fclose(out);
		return true;
	}

private:
	std::string file_path_prefix_;
	std::string folder_;
	std::vector<int> steps_;

};

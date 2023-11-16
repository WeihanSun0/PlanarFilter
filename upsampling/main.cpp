#include <upsampling.h>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <chrono>

void depth2pc(const cv::Mat& depth, const cv::Mat& K, cv::Mat& pc) {
	pc.create(depth.size(), CV_32FC3);

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);

	for (int y = 0; y < depth.rows; ++y) {
		for (int x = 0; x < depth.cols; ++x) {
			float z = depth.at<float>(y, x);
			pc.at<cv::Vec3f>(y, x)[0] = (x - cx) * z / fx;
			pc.at<cv::Vec3f>(y, x)[1] = (y - cy) * z / fy;
			pc.at<cv::Vec3f>(y, x)[2] = z;
		}
	}
}

bool isBreak = false;
void  KeyboardViz3d(const cv::viz::KeyboardEvent& w, void* t)
{
	cv::viz::Viz3d* fen = (cv::viz::Viz3d*)t;
	if (w.action) {
		std::cout << "input = " << w.code << std::endl;
	}

	if (w.code == 'q')
		isBreak = true;
}

int main(int argc, char* argv[]) {
	std::cout << "begin" << std::endl;
	cv::Rect roi(0, 0, 320, 240);
	float scale = 1.0f; 

	// class 定義
	upsampling dc;

	int index = 0;
	bool pause_mode = false;

	std::chrono::system_clock::time_point  start, end;
		// データ用意
	char fn_depth[40], fn_rgb[40];
	sprintf_s(fn_rgb, "rgb.png");
	sprintf_s(fn_depth, "sp.tiff");
	cv::Mat org_guide = cv::imread(fn_rgb);
	cv::Mat sp_depth = cv::imread(fn_depth, cv::IMREAD_UNCHANGED);//1x3x576, 1x576x3, 3x24x24に対応
	if (org_guide.empty() || sp_depth.empty())
		exit(1);
	// main part
	cv::Mat guide;
	start = std::chrono::system_clock::now();
	cv::resize(org_guide, guide, cv::Size(), 0.25, 0.25);
	// test
	cv::Mat dense_depth, confidence;
	dc.run2(guide, sp_depth*1000, roi, dense_depth, confidence);
	end = std::chrono::system_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "total time = " << elapsed << " [ms]" << std::endl;
	cv::imwrite("dense_depth.tiff", dense_depth);
	cv::imwrite("confidence.tiff", confidence);
	std::cout << "end" << std::endl;
	return 0;
}
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudalegacy.hpp"
#include <algorithm>
using namespace std;
using namespace cv;
using namespace cv::cuda;
inline uint getFirstIndex(uchar, uchar, uchar);


Scalar hsv_min(0,215,170);
Scalar hsv_max(160,200,200);
const int sizeX = 640;
const int sizeY = 480;
const int minArea = 550;
const int minSolidity = 0.65;
const double expectedAspectRation = 1.6;
const double aspectRatioTolerance = 0.6;
const Mat camera_matrix = (cv::Mat_<double>(3,3) << 786.42, 0, 297.35, 0 , 780.45, 214.74, 0, 0, 1);
const Mat dist_coeffs = (cv::Mat_<double>(5,1) <<  2.02296730e-01, -3.61888606e00,  -9.66524854e-03, -8.83399450e-03, 1.41721964e+01);
const Mat model_points = (cv::Mat_<Point3d>(8,1) <<  Point3d(0,0,0), Point3d(0,0,0),  Point3d(0,0,0), Point3d(0,0,0), Point3d(0,0,0),Point3d(0,0,0),Point3d(0,0,0),Point3d(0,0,0));

uchar *LUMBGR2HSV;
uchar *d_LUMBGR2HSV;
__global__
void kernelconvert(uchar *LUT)
{
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
	uint k = (blockIdx.z * blockDim.z) + threadIdx.z;
	if (i < 256 && j < 256 && k < 256) {
		uchar _b = i;
		uchar _g = j;
		uchar _r = k;
		float b = (float)_b / 255.0;
		float g = (float)_g / 255.0;
		float r = (float)_r / 255.0;
		float h, s, v;
		float _min = min(min(b, g), r);
		v = max(max(b, g), r);
		float chroma = v - _min;
		if (v != 0)
			s = chroma / v; // s
		else {
			s = 0;
			h = -1;
			return;
		}
		if (r == v)
			h = (g - b) / chroma;
		else if (g == v)
			h = 2 + (b - r) / chroma;
		else
			h = 4 + (r - g) / chroma;
		h *= 30;
		if (h < 0)	h += 180;
		s *= 255;
		v *= 255;
		uint index = 3 * 256 * 256 * i + 256 * 3 * j + 3 * k;
		LUT[index] = (uchar)h;
		LUT[index + 1] = (uchar)s; //height, width  Saturation
		LUT[index + 2] = (uchar)v; //height, width  Value
	}
}
__global__
void kernelSwap(PtrStepSz<uchar3> src, PtrStepSz<uchar3>  dst, uchar *LUT) {
	uint x = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= src.cols || y >= src.rows) return;
	uchar3 v = src(y,x);
	uint index = 3 * 256 * 256 * v.x + 256 * 3 * v.y + 3 * v.z;
	dst(y,x).x = LUT[index];
	dst(y,x).y = LUT[index+1];
	dst(y,x).z = LUT[index+2];
}
inline uint getFirstIndex(uchar b, uchar g, uchar r) {
	return 3 * 256 * 256 * b + 256 * 3 * g + 3 * r;
}
void initializeLUM() {
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc((void **)&LUMBGR2HSV, 256*256*256*3, cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&d_LUMBGR2HSV, (void *) LUMBGR2HSV, 0);
	dim3 threads_per_block(8, 8,8);
	dim3 numBlocks(32,32,32);
	kernelconvert << <numBlocks, threads_per_block >> >(d_LUMBGR2HSV);
}
void BGR2HSV_LUM(GpuMat src, GpuMat dst) {
	const int m = 32;
	int numRows = src.rows, numCols = src.cols;
	if (numRows == 0 || numCols == 0) return;
	// Attention! Cols Vs. Rows are reversed
	const dim3 gridSize(ceil((float)numCols / m), ceil((float)numRows / m), 1);
	const dim3 blockSize(m, m, 1);
	kernelSwap << <gridSize, blockSize >> >(src, dst, d_LUMBGR2HSV);
}
__global__ void inRange_kernel(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSzb dst,
                               int lbc0, int ubc0, int lbc1, int ubc1, int lbc2, int ubc2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= src.cols || y >= src.rows) return;

  uchar3 v = src(y, x);
  if (v.x >= lbc0 && v.x <= ubc0 && v.y >= lbc1 && v.y <= ubc1 && v.z >= lbc2 && v.z <= ubc2)
    dst(y, x) = 255;
  else
    dst(y, x) = 0;
}
void inRange_gpu(cv::cuda::GpuMat &src, cv::Scalar &lowerb, cv::Scalar &upperb,
                 cv::cuda::GpuMat &dst) {
  const int m = 32;
  int numRows = src.rows, numCols = src.cols;
  if (numRows == 0 || numCols == 0) return;
  // Attention! Cols Vs. Rows are reversed
  const dim3 gridSize(ceil((float)numCols / m), ceil((float)numRows / m), 1);
  const dim3 blockSize(m, m, 1);
  inRange_kernel<<<gridSize, blockSize>>>(src, dst, lowerb[0], upperb[0], lowerb[1], upperb[1],
                                          lowerb[2], upperb[2]);
}
Mat getHsvMasked(Mat frame)	{
	GpuMat frame_gpu, mask_gpu;
	frame_gpu.upload(frame);
	BGR2HSV_LUM(frame_gpu, frame_gpu);
	mask_gpu.create(frame_gpu.rows, frame_gpu.cols, CV_8U);
	inRange_gpu(frame_gpu, hsv_min, hsv_max, mask_gpu);
	Mat mask(mask_gpu);
	return mask;
}
vector<RotatedRect> getPotentialTargets(Mat mask)	{
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(mask,contours,hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	vector<RotatedRect> targets;
	cout << "Contours Found: "<<contours.size() << "\n";
	for(int i = 0; i < contours.size(); i++)	{
		int area = contourArea(contours[i]);
		if(area > minArea) 	{
			RotatedRect rect = minAreaRect(contours[i]);
			//use shorter side as width when calculating aspect ratio
			int height = (rect.size.height > rect.size.width) ? rect.size.height :  rect.size.width;
			int width = (rect.size.height > rect.size.width) ? rect.size.height :  rect.size.width;
			if(abs((float)width/(float)height - expectedAspectRation) < aspectRatioTolerance)	{
				vector<Point>  hull;
				convexHull(contours[i], hull);
				int hull_area = contourArea(hull);
				float solidity = float(area)/hull_area;
				if(solidity > minSolidity)	{
					targets.push_back(rect);
				}
			}
		}
	}
	sort(targets.begin(), targets.end(), [](const RotatedRect& a, const RotatedRect& b)	{
		return a.center.x < b.center.x;
	});
	return targets;
}
int getStripType(RotatedRect strip)	{
	if(strip.size.height > strip.size.width)	{
		return 1;
	}	else {
		return 2;
	}
}
class VisionTarget {
	public:
		RotatedRect left;
		RotatedRect right;
		int getCenterX() {
			return (left.center.x + right.center.x)/2;
		}
		vector<Point2d> leftTargetPointsClockwiseFromLowest()	{
			vector<Point2d> points;
			Point2f pts[4];
			left.points(pts);
			for (int i = 0 ; i < 4 ; i++)
			{
  			points.push_back((Point2d)pts[i]);
			}
			return points;
		}
		vector<Point2d> rightTargetPointsClockwiseFromLowest()	{
			vector<Point2d> points;
			Point2f pts[4];
			right.points(pts);
			for (int i = 0 ; i < 4 ; i++)
			{
				points.push_back((Point2d)pts[i]);
			}
			return points;
		}
		vector<Point2d> eightPointImageDescriptor()	{
				vector<Point2d> points;
				vector<Point2d> leftPoints = leftTargetPointsClockwiseFromLowest();
				vector<Point2d> rightPoints = rightTargetPointsClockwiseFromLowest();
				points.reserve(8);
				points.insert(points.end(), leftPoints.begin(), leftPoints.end());
				points.insert(points.end(), rightPoints.begin(), rightPoints.end());
				return points;
		}
};
VisionTarget getVisionTarget(vector<RotatedRect> potentialTargets)	{
	vector<VisionTarget> targets;
	/*target.TargetType = 0;
	if(potentialTargets.size() == 1)	{
		if(getStripType(potentialTargets[0]) == 1)	{
			target.TargetType = 1;
			target.right = potentialTargets[0];
		}	else {
			target.TargetType = 2;
			target.right = potentialTargets[0];
		}
	}*/
	if(potentialTargets.size() > 1)	{
		for(int i = 0; i < potentialTargets.size()-1; i++)	{
			if(getStripType(potentialTargets[i]) == 2 && getStripType(potentialTargets[i+1]) == 1) {
				VisionTarget temp;
				temp.right = potentialTargets[i+1];
				temp.left = potentialTargets[i];
				targets.push_back(temp);
			}
		}
		sort(targets.begin(), targets.end(), [](VisionTarget a, VisionTarget b)	{
			return abs(a.getCenterX()-sizeX/2) < abs(b.getCenterX()-sizeX/2);
		});
	}
	return targets[0];
}



int main(int argc, char** argv)
{
	setDevice(0);
	initializeLUM();
	VideoCapture capture("/dev/video1");
	VideoWriter video;
	video.open("appsrc ! videoconvert ! video/x-raw, format=(string)I420, width=(int)640, height=(int)480 ! omxh264enc bitrate=600000 ! video/x-h264, stream-format=(string)byte-stream ! h264parse ! rtph264pay ! udpsink host=10.0.0.60 port=5000 sync=true ", 0, 30, cv::Size(1280, 720), true);
	capture.set(CAP_PROP_AUTOFOCUS, 0);
	capture.set(CAP_PROP_FRAME_WIDTH, sizeX);
	capture.set(CAP_PROP_FRAME_HEIGHT, sizeY);
	for (; ; )
	{
		Mat frame, mask;
		capture.read(frame);
		if (frame.empty())	{
			break;
		}
		mask = getHsvMasked(frame);
		vector<RotatedRect> targets = getPotentialTargets(mask);
		cout << "Targets Found: "<<targets.size() << "\n";
		if(targets.size() <= 1) continue;
		VisionTarget target = getVisionTarget(targets);
		vector<cv::Point2d> image_points;
		Mat image_points_matrix = Mat(image_points);
		Mat rotation_vector; // Rotation in axis-angle form
    Mat translation_vector;
		image_points = target.eightPointImageDescriptor();
		cv::cuda::solvePnPRansac(model_points,image_points_matrix,camera_matrix,dist_coeffs,rotation_vector, translation_vector);
		video.write(frame);
		cout << "Rotation Vector " << endl << rotation_vector << endl;
    cout << "Translation Vector" << endl << translation_vector << endl;
	}
}

#include <stdio.h>
#include "networktables/NetworkTable.h" //networktables
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/cudalegacy.hpp"
#include <algorithm>
#include <thread>
#include <chrono>
#include <mutex>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>

#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;
inline uint getFirstIndex(uchar, uchar, uchar);

shared_ptr<NetworkTable> myNetworkTable; //our networktable for reading/writing
string netTableAddress = "10.0.0.60";
std::mutex frame_mutex;  // protects frame //TODO

const int sizeX = 640;
const int sizeY = 480;
const int fps = 15;
//TODO: String formatter
const string STREAM_STRING = "appsrc ! videoconvert ! video/x-raw, format=(string)I420, width=(int)640, height=(int)480 ! omxh264enc bitrate=600000 ! video/x-h264, stream-format=(string)byte-stream ! h264parse ! rtph264pay ! udpsink host=10.0.0.60 port=5801 sync=true ";
const string DEBUG_STRING = "appsrc ! videoconvert ! video/x-raw, format=(string)I420, width=(int)640, height=(int)480 ! omxh264enc bitrate=600000 ! video/x-h264, stream-format=(string)byte-stream ! h264parse ! rtph264pay ! udpsink host=10.0.0.60 port=5802 sync=true ";
VideoWriter debug;


const Mat camera_matrix = (cv::Mat_<float>(3,3) << 786.42, 0, 297.35, 0 , 780.45, 214.74, 0, 0, 1);
//const Mat dist_coeffs = (cv::Mat_<float>(1,5) <<  2.02296730e-01, -3.61888606e00,  -9.66524854e-03, -8.83399450e-03, 1.41721964e+01);
const Mat dist_coeffs = (cv::Mat_<float>(1,5) <<  0, 0,  0, 0, 0);
const Mat model_points = (cv::Mat_<Point3f>(1,8) <<  Point3d(-4.38,-5.32,0),  Point3d(-6.313,-4.819,0), Point3d(-5.936,0.5,0),  Point3d(-4,0,0), Point3d(4.377,-5.32,0), Point3d(4,0,0),Point3d(5.936,0.5,0),Point3d(6.313,-4.82,0));

Scalar hsv_min(0,0,37);
Scalar hsv_max(180,255,255);
const int minArea = 200;
const int minSolidity = 0.85;
const double expectedAspectRation = 3.5;
const double aspectRatioTolerance = 2;

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
	//Mat inHSV(frame_gpu);
	//imshow("HSV", inHSV);
	inRange_gpu(frame_gpu, hsv_min, hsv_max, mask_gpu);
	Mat mask(mask_gpu);
	//imshow("threshold",mask);
	//waitKey(1);
	return mask;
}
vector<RotatedRect> getPotentialTargets(Mat mask)	{
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(mask,contours,hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	vector<RotatedRect> targets;
	//cout << "Contours Found: "<<contours.size() << "\n";
	for(int i = 0; i < contours.size(); i++)	{
		int area = contourArea(contours[i]);
		if(area > minArea) 	{
			//cout << "Area " << area << "\n";
			RotatedRect rect = minAreaRect(contours[i]);
			//use shorter side as width when calculating aspect ratio
			int height = (rect.size.height > rect.size.width) ? rect.size.height :  rect.size.width;
			int width = (rect.size.height > rect.size.width) ? rect.size.width :  rect.size.height;
			if(abs((float)((float)height/(float)width) - expectedAspectRation) < aspectRatioTolerance)	{
				vector<Point>  hull;
				convexHull(contours[i], hull);
				int hull_area = contourArea(hull);
				float solidity = float(area)/hull_area;
				if(solidity > minSolidity)	{
					//cout << "Center of Potential Target: " << rect.center.x << ", " << rect.center.y << " Aspect " << (float)((float)height/(float)width) <<  "\n";
					//cout << "solidity " << solidity << "\n";
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
		int targetType;
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
	VisionTarget Target;
	Target.targetType = 0;
	if(potentialTargets.size() > 1)	{
		for(int i = 0; i < potentialTargets.size()-1; i++)	{
			if(getStripType(potentialTargets[i]) == 2 && getStripType(potentialTargets[i+1]) == 1) {
				VisionTarget temp;
				temp.right = potentialTargets[i+1];
				temp.left = potentialTargets[i];
				targets.push_back(temp);
			}
		}
		//do this in O(n)
		sort(targets.begin(), targets.end(), [](VisionTarget a, VisionTarget b)	{
			return abs(a.getCenterX()-sizeX/2) < abs(b.getCenterX()-sizeX/2);
		});
		if(targets.size() > 0)	{
			Target = targets[0];
			Target.targetType = 1;
		}
	}
	return Target;
}

vector<cv::Point2d> getImagePointsFromFrame(Mat* frame)	{
	Mat mask;
	Scalar color(0,0,255);
	vector<cv::Point2d> image_points;
	mask = getHsvMasked(*frame);
	vector<RotatedRect> targets = getPotentialTargets(mask);
	if(targets.size() >= 1) {
		VisionTarget target = getVisionTarget(targets);
		if(target.targetType == 1) {
			image_points = target.eightPointImageDescriptor();
			/*
			for(Point2f p : image_points)	{
				circle(*frame, p, 5,color,5,LINE_8);
			}*/
		}
	}
	//debug.write(*frame);
	return image_points;
}

Vec3d getEulerAngles(Mat rotation_vector){
		Mat rotation3x3;
		Vec3d eulerAngles;
		Rodrigues(rotation_vector, rotation3x3);
    Mat cameraMatrix,rotMatrix,transVect,rotMatrixX,rotMatrixY,rotMatrixZ;

    double* _r = rotation3x3.ptr<double>();

    double projMatrix[12] = {_r[0],_r[1],_r[2],0,
                          _r[3],_r[4],_r[5],0,
                          _r[6],_r[7],_r[8],0};

    decomposeProjectionMatrix( Mat(3,4,CV_64FC1,projMatrix),
                               cameraMatrix,
                               rotMatrix,
                               transVect,
                               rotMatrixX,
                               rotMatrixY,
                               rotMatrixZ,
                               eulerAngles);
		return eulerAngles;
}

void getRotationAndTranslationVectors(Mat* frame,Mat* rotation_vector,Mat* translation_vector, bool* newVector)	{
	vector<cv::Point2d> image_points;
	image_points = getImagePointsFromFrame(frame);
	if(image_points.size() != 8) {
			*newVector = false;
			return;
	}
	Mat image_points_matrix = Mat(image_points);
	dist_coeffs.convertTo(dist_coeffs,CV_32F);
	*newVector = cv::solvePnP(model_points,image_points_matrix,camera_matrix,dist_coeffs,*rotation_vector, *translation_vector, false,  SOLVEPNP_ITERATIVE);
}

void processFrameThread(Mat* frame,Mat* rotation_vector,Mat* translation_vector, bool* newImage, bool* newVector)	{
	for(;	; )	{
		if(*newImage == false) continue;
		getRotationAndTranslationVectors(frame,rotation_vector,translation_vector, newVector);
		//cout << "Frame Processed\n";
		if(*newVector)	{
			Vec3d orientation = getEulerAngles(*rotation_vector);
			string s = to_string((*translation_vector).at<double>(2,0)) + ";" + to_string((*translation_vector).at<double>(1,0)) + ";" + to_string((*translation_vector).at<double>(0,0)) + ";" +  to_string(orientation[1]) + ";\n";
			cout << s;
		}
		*newImage = false;
	}
}

void printInfo()	{
	NetworkTable::SetClientMode();
	//NetworkTable::SetDSClientEnabled(false);
	NetworkTable::SetIPAddress(llvm::StringRef(netTableAddress));
	NetworkTable::Initialize();
	myNetworkTable = NetworkTable::GetTable("JetsonData");
}
int main(int argc, char** argv)
{
	setDevice(0);
	initializeLUM();
	char setting_script[100];
	sprintf (setting_script, "bash good_settings.sh %d", 1);
	system (setting_script);
	VideoCapture capture("/dev/video1");
	//VideoWriter video;
	Mat rotation_vector; // Rotation in axis-angle form
	Mat translation_vector;
	Mat frame;
	bool newImage = false;
	bool newVector = false;
	//video.open(STREAM_STRING, 0, 30, cv::Size(sizeX, sizeY), true);
	//debug.open(DEBUG_STRING, 0,30,cv::Size(sizeX, sizeY), true);
	capture.set(CAP_PROP_FRAME_WIDTH, sizeX);
	capture.set(CAP_PROP_FRAME_HEIGHT, sizeY);
	capture.set(CAP_PROP_FPS, fps);
	//thread print (printInfo);
	thread process (processFrameThread,&frame,&rotation_vector,&translation_vector,&newImage, &newVector);
	int64_t lastImageSentTime = 0;
	for (; ; )
	{
		capture.read(frame);
		if (frame.empty())	{
			break;
		}
		if(newVector)	{
			Vec3d orientation = getEulerAngles(rotation_vector);
			//Z, Y, X, yaw
			/*
			myNetworkTable -> PutNumber ("Z Displacement", (translation_vector).at<double>(2,0));
	 		myNetworkTable -> PutNumber ("Y Displacement", (translation_vector).at<double>(1,0));
	 		myNetworkTable -> PutNumber ("X Displacement", (translation_vector).at<double>(0,0));
			myNetworkTable -> PutNumber ("Yaw", orientation[1]);
	 		myNetworkTable -> Flush();
			*/
			newVector = false;
		}
		newImage = true;
	}
}

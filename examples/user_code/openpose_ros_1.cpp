// C++ std library dependencies
#include <atomic> // std::atomic
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <cstdio> // sscanf
#include <string> // std::string
#include <thread> // std::this_thread
#include <vector> // std::vector
// OpenCV dependencies
#include <opencv2/core/core.hpp> // cv::Mat & cv::Size
// Other 3rdpary depencencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h>  // google::InitGoogleLogging, CHECK, CHECK_EQ, LOG, VLOG, ...
#include <openpose/headers.hpp>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

//ROS add///////////////////
#include <iostream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
cv::Mat image_input;
ros::Publisher pcl_pub;
#include <unistd.h>
///////////////////////////

// Gflags in the command line terminal. Check all the options by adding the flag `--help`, e.g. `rtpose.bin --help`.
// Note: This command will show you flags for several files. Check only the flags for the file you are checking. E.g. for `rtpose`, look for `Flags from examples/openpose/rtpose.cpp:`.
// Debugging
DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while 255 will not output any."
							   " Current OpenPose library messages are in the range 0-4: 1 for low priority messages and 4 for important ones.");
// Producer
//DEFINE_string(image_path,   "examples/media/COCO_val2014_000000000192.jpg", "Process the desired image.");
// OpenPose
DEFINE_string(model_pose, "COCO", "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder, "models/", "Folder where the pose models (COCO and MPI) are located.");
DEFINE_string(net_resolution, "656x368", "Multiples of 16.");

//ROS Add
DEFINE_string(resolution, "800x600", "The image resolution (display). Use \"-1x-1\" to force the program to use the default images resolution.");

DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you want to change the initial scale, "
							  "you actually want to multiply the `net_resolution` by your desired initial scale.");
DEFINE_int32(num_scales, 1, "Number of scales to average.");
// OpenPose Rendering
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will hide it.");

op::PoseModel gflagToPoseModel(const std::string &poseModeString)
{
	op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	if (poseModeString == "COCO")
		return op::PoseModel::COCO_18;
	else if (poseModeString == "MPI")
		return op::PoseModel::MPI_15;
	else if (poseModeString == "MPI_4_layers")
		return op::PoseModel::MPI_15_4;
	else
	{
		op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
		return op::PoseModel::COCO_18;
	}
}

// Google flags into program variables
std::tuple<cv::Size, cv::Size, cv::Size, op::PoseModel> gflagsToOpParameters()
{
	op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	// outputSize
	cv::Size outputSize;
	auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.width, &outputSize.height);
	op::checkE(nRead, 2, "Error, resolution format (" + FLAGS_resolution + ") invalid, should be e.g., 960x540 ", __LINE__, __FUNCTION__, __FILE__);
	// netInputSize
	cv::Size netInputSize;
	nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.width, &netInputSize.height);
	op::checkE(nRead, 2, "Error, net resolution format (" + FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
	// netOutputSize
	const auto netOutputSize = netInputSize;
	// poseModel
	const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
	// Check no contradictory flags enabled
	if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
		op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
	if (FLAGS_scale_gap <= 0. && FLAGS_num_scales > 1)
		op::error("Uncompatible flag configuration: scale_gap must be greater than 0 or num_scales = 1.", __LINE__, __FUNCTION__, __FILE__);
	// Logging and return result
	op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	return std::make_tuple(outputSize, netInputSize, netOutputSize, poseModel);
}

int openPoseTutorialPose1()
{
	op::log("OpenPose + ROS Tutorial - Example 1.", op::Priority::Max);
	// ------------------------- INITIALIZATION -------------------------
	// Step 1 - Set logging level
	// - 0 will output all the logging messages
	// - 255 will output nothing
	op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
	op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
	// Step 2 - Read Google flags (user defined configuration)
	cv::Size outputSize;
	cv::Size netInputSize;
	cv::Size netOutputSize;
	op::PoseModel poseModel;
	std::tie(outputSize, netInputSize, netOutputSize, poseModel) = gflagsToOpParameters();
	// Step 3 - Initialize all required classes
	op::CvMatToOpInput cvMatToOpInput{netInputSize, FLAGS_num_scales, (float)FLAGS_scale_gap};
	op::CvMatToOpOutput cvMatToOpOutput{outputSize};
	op::PoseExtractorCaffe poseExtractorCaffe{netInputSize, netOutputSize, outputSize, FLAGS_num_scales, (float)FLAGS_scale_gap, poseModel,
											  FLAGS_model_folder, FLAGS_num_gpu_start};
	op::PoseRenderer poseRenderer{netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_alpha_pose};
	op::OpOutputToCvMat opOutputToCvMat{outputSize};
	const cv::Size windowedSize = outputSize;
	//  op::FrameDisplayer frameDisplayer{windowedSize, "OpenPose Tutorial - Example 1"};
	// Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
	poseExtractorCaffe.initializationOnThread();
	poseRenderer.initializationOnThread();

	for (;;)
	{
		///////////ROS add
		ros::spinOnce();
		///		
		// ------------------------- POSE ESTIMATION AND RENDERING -------------------------
		// Step 1 - Read and load image, error if empty (possibly wrong path)
		cv::Mat inputImage = image_input;
		if (inputImage.empty())
			break;
		// Step 2 - Format input image to OpenPose input and output formats
		const auto netInputArray = cvMatToOpInput.format(inputImage);
		double scaleInputToOutput;
		op::Array<float> outputArray;
		std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(inputImage);
		// Step 3 - Estimate poseKeyPoints
		poseExtractorCaffe.forwardPass(netInputArray, inputImage.size());
		const auto poseKeyPoints = poseExtractorCaffe.getPoseKeyPoints();
		// Step 4 - Render poseKeyPoints
		poseRenderer.renderPose(outputArray, poseKeyPoints);
		// Step 5 - OpenPose output format to cv::Mat
		auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

		//POSE_COCO_BODY_PARTS{{0, "Nose"}, {1, "Neck"}, {2, "RShoulder"}, {3, "RElbow"}, {4, "RWrist"},
		//		{5, "LShoulder"}, {6, "LElbow"}, {7, "LWrist"}, {8, "RHip"}, {9, "RKnee"},
		//		{10, "RAnkle"}, {11, "LHip"}, {12, "LKnee"}, {13, "LAnkle"}, {14, "REye"},
		//		{15, "LEye"}, {16, "REar"}, {17, "LEar"}, {18, "Bkg"},}


		pcl::PointCloud<pcl::PointXYZ> cloud;
		sensor_msgs::PointCloud2 output;
		std::cout<<poseKeyPoints.getVolume()<<std::endl;
		int coco_size=poseKeyPoints.getVolume()/3;

		cloud.points.resize(coco_size);
		int poseKeyPoints_size=0;
		for (int i = 0; i < coco_size; i++){
			cloud.points[i].x = poseKeyPoints[poseKeyPoints_size++]/100.0;
			cloud.points[i].y = poseKeyPoints[poseKeyPoints_size++]/100.0;
			cloud.points[i].z = poseKeyPoints[poseKeyPoints_size++]/100.0;
		}

		pcl::toROSMsg(cloud, output);
		output.header.frame_id = "camera_link";
		pcl_pub.publish(output);

		// ------------------------- SHOWING RESULT AND CLOSING -------------------------
		// Step 1 - Show results
		cv::imshow("outputImage", outputImage);
		int key = cv::waitKey(30);
		if (key == 'q')
			break;

		// Step 2 - Logging information message
		//op::log("Example 1 successfully finished.", op::Priority::Max);
		// Return successful message
	}

	return 0;
}
///////////ROS add
void imageCallback(const sensor_msgs::ImageConstPtr &msg, const sensor_msgs::CameraInfoConstPtr &info)
{
	cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
	image_input = image.clone();
}
///////////////////
int main(int argc, char *argv[])
{

	ros::init(argc, argv, "image_openpose");
	ros::NodeHandle nh;

	pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("skeleton_output", 1);
	image_transport::ImageTransport it(nh);
	image_transport::CameraSubscriber sub;

	sub = it.subscribeCamera("camera_raw", 1, imageCallback);
	//sleep(1);
	//ros::spinOnce();
	std::cout << "InitGoogleLogging" << std::endl;

	google::InitGoogleLogging("openPoseTutorialWrapper2");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Running openPoseTutorialWrapper2
	return openPoseTutorialPose1();
}

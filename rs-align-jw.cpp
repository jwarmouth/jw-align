// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp>
#include "example-imgui.hpp"
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include "../cv-helpers.hpp"    // Helper functions for conversions between RealSense and OpenCV

// https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <map>
#include <string>
#include <thread>
#include <atomic>

#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>

#include <imgui.h>
#include "imgui_impl_glfw.h"

#include <ctime>
#include <experimental/filesystem>

// 3rd party header for writing png files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace cv;
using namespace rs2;

// Pseudocode - TODO
// X read color_frame and depth_frame
// X post-process and clean depth_frame
// X align depth_frame to color_frame
// X create cv_frame - just white/black data
// X create matted_frame (color_frame using aligned_corrected_depth_frame as mat)
// X flip like a mirror
// resized_frame is resized matted_frame (128x128?)
// queue resized_frame into sprite_sheet
// once finished, save sprite_sheet
// trigger to create new gnome
// load sprite_sheet for each new gnome
// loop each gnome's sprite_sheet
// gravity system
// Remove Floor from depth map (use point cloud?)
// Crop top and/or bottom of depth image
// UI system to fine-tune on site?
// Rotate https://note.nkmk.me/en/python-opencv-numpy-rotate-flip/
// https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance#section-make-sure-image-is-properly-exposed
// https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/BKMs_Tuning_RealSense_D4xx_Cam.pdf
// https://github.com/IntelRealSense/librealsense/wiki/D400-Series-Visual-Presets

// HOW TO?
// X save a png
// X Allocate CV images with masks
// X resize a cv Mat
// X resize a cv::mat & display in a rect
// load an image in cv
// save a cv as a png image
// resize an image into a... texture? file?
// draw or render image into a sprite sheet
// Use FBOs?
// Simple gravity system
// keep track of frame rate / deltaTime
// Build for Pi 4?
// X Buy an Intel NUC

//void render_slider(rect location, float& clipping_dist);

//create Background Subtractor objects
//Ptr<BackgroundSubtractor> background_subtractor = createBackgroundSubtractorKNN(30, 800.0f, true);
Ptr<BackgroundSubtractor> background_subtractor = createBackgroundSubtractorMOG2(500, 32, true);
//Ptr<BackgroundSubtractor> background_subtractor = createBackgroundSubtractorMOG2();
//Mat foreground_mask;

//void remove_background(rs2::video_frame& other, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_min, float clipping_max);
float get_depth_scale(rs2::device dev);
rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);
void overlayImage(Mat& background, Mat& foreground, Mat& output, const Point& location);


int main(int argc, char * argv[]) try
{
    // Create and initialize GUI related objects
    // window app(1280, 720, "Lilliput RS Test"); // Simple window handling
    // ImGui_ImplGlfw_Init(app, false);      // ImGui library intializition
    // texture renderer;                     // Helper for rendering images

    bool alive = true;

    // Read GnomesDirectory from gnomesDir.txt
    // Declare Gnomes Folder
    std::string gnomes_folder = "C:/Gnomes/";


    // SET UP REALSENSE CAMERA
    rs2::pipeline pipe;
    rs2::pipeline_profile profile;
    rs2::device device;
    rs2::colorizer color_map;
    rs2::pointcloud pc;
    rs2::points points;
    rs2_stream color_stream;
    rs2_stream depth_stream;
    //rs2::pointcloud original_pc;
    //rs2::pointcloud filtered_pc;

    // Configure the streams
    rs2::config config;
    config.enable_stream(RS2_STREAM_COLOR, 1280, 0, RS2_FORMAT_RGB8, 30);
    config.enable_stream(RS2_STREAM_DEPTH, 848, 0, RS2_FORMAT_Z16, 30);
    //cfg.resolve(pipe);
    //cfg.enable_record_to_file("abc.bag");

    profile = pipe.start(config);
    device = profile.get_device();

    color_stream = RS2_STREAM_COLOR;
    depth_stream = RS2_STREAM_DEPTH;
    rs2_stream align_to = color_stream; // find_stream_to_align(profile.get_streams());
    rs2::align align(align_to);
    // https://github.com/IntelRealSense/librealsense/issues/1706
    // https://github.com/IntelRealSense/librealsense/issues/5482
    //auto extrinsics = depth_stream.get_extrinsics_to(color_stream); // https://community.intel.com/t5/Items-with-no-label/How-to-project-points-to-pixel-Librealsense-SDK-2-0/td-p/479189
    // Extrinsics https://github.com/IntelRealSense/librealsense/issues/5482


    // Define variables for controlling the distance to clip
    float depth_clipping_min = 1.5f;
    float depth_clipping_max = 2.5f;
    float depth_scale;
    depth_scale = get_depth_scale(device);


    // POST-PROCESSING FILTERS
    // Depth Frame > Decimation > Depth2Disparity > Spatial > Temporal >> Disparity2Depth > Hole Filling > Filtered Depth
    // https://dev.intelrealsense.com/docs/post-processing-filters
    // https://dev.intelrealsense.com/docs/depth-post-processing
    // Using align with post-processing https://github.com/IntelRealSense/librealsense/issues/1207

    // Declare filters
    rs2::decimation_filter decimation_filter;  // Decimation - reduces depth frame density, closing small holes and speeding-up the algorithm
    rs2::disparity_transform depth2disparity;  // To make sure far-away objects are filtered proportionally
    rs2::spatial_filter spatial_filter;    // Spatial    - edge-preserving spatial smoothing
    rs2::temporal_filter temporal_filter;   // Temporal   - reduces temporal noise
    rs2::disparity_transform disparity2depth(false);
    rs2::threshold_filter threshold_filter;   // Threshold  - removes values outside recommended range
    rs2::hole_filling_filter hole_filter; // Hole Filling

    // Configure filter parameters
    decimation_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    spatial_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.55f);
    spatial_filter.set_option(RS2_OPTION_HOLES_FILL, 1);
    threshold_filter.set_option(RS2_OPTION_MIN_DISTANCE, depth_clipping_min);
    threshold_filter.set_option(RS2_OPTION_MAX_DISTANCE, depth_clipping_max);
    //hole_filter.set_option(RS2_OPTION_HOLES_FILL, 1);

    // For CV processing
    const auto window_name = "CV Frame";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    //namedWindow("CV Mask", WINDOW_AUTOSIZE);
    //*
    //cv::Mat color_cv_frame;
    //cv::Mat depth_cv_frame;
    cv::Mat depth_cv_mask;
    cv::Mat final_cv_frame;
    int thresh_min_value = 150;
    int thresh_max_value = 255;

    //std::string gnome_path("C:\Gnomes\gnome_0100.png");
    //cv::Mat gnome_image = imread(gnome_path, -1);

    // Use StructuringElement for erode / dilate operations
    auto gen_element = [](int erosion_size)
    {
        return getStructuringElement(MORPH_RECT,
            Size(erosion_size + 1, erosion_size + 1),
            Point(erosion_size, erosion_size));
    };

    const int erosion_size = 3;
    auto erode_less = gen_element(erosion_size);
    auto erode_more = gen_element(erosion_size * 2);

    // Take grayscale image, perfors threshold on it, close small holes and erode the white area
    auto create_mask_from_depth = [&](Mat& depth, int thresh, ThresholdTypes type)
    {
        threshold(depth, depth, thresh, 255, type);
        dilate(depth, depth, erode_less);
        erode(depth, depth, erode_more);
    };
    // */


    rs2::frame_queue postprocessed_frames;

    // POST-PROCESSING THREAD
    //https://github.com/IntelRealSense/librealsense/blob/master/doc/frame_lifetime.md#frames-and-threads

    std::thread video_processing_thread([&]() {
        // In order to generate new composite frames, we have to wrap the processing code in a lambda
        rs2::processing_block frame_processor([&](rs2::frameset data, rs2::frame_source& source) // Frame pool that can allocate new frames
            {
                // Apply post-processing filters to depth frame
                rs2::depth_frame depth = data.get_depth_frame();
                depth = decimation_filter.process(depth);
                depth = threshold_filter.process(depth);
                depth = depth2disparity.process(depth);
                depth = spatial_filter.process(depth);
                depth = temporal_filter.process(depth);
                depth = disparity2depth.process(depth);
                depth = hole_filter.process(depth);

                // convert depth map to point cloud
                //points = pc.calculate(depth_frame);


                // project point cloud back to depth frame

                // Remove Ground Plane using brute force
                /*
                //https://medium.com/@rivindum.15/human-silhouette-extraction-from-depth-videos-b84d2ac77bd4
                auto height = depth.get_height();
                auto width = depth.get_width();
                float threshold = 5.f;
                int cy = int(width / 2);        // center y
                int fy = 580;                   //focal length y
                int z = 5000;                     //initial min depth
                int[height] croppedRow; = img_array[-1][np.nonzero(img_array[-1])]; // UH...
                if (croppedRow.length) > 0)
                    z = stats.mode(croppedRow).mode[0];      //initial floor depth
                else
                    z = 75;
                
                float Y = z * (height - cy) / fy;                         //actual y of floor
                Y = Y - threshold;
                for (int i = height - 1; i > cy - 1; i -= 1)
                {
                    float z1 = (Y * fy) / ((i + 1) - cy);                   //depth of Y height
                    for (int j = 0; j < width; j++)
                    {
                        if (img_array[i][j] > z1) :
                            Img_array[i][j] = 0
                    }
                                // output, __ = normalizeImage(img_array)
                }
                */


                // Remove points beyond threshold

                auto color = data.get_color_frame();
                rs2::frameset combined = source.allocate_composite_frame({ depth, color });
                combined = align.process(combined);
                source.frame_ready(combined);
            });
        // Push the results of frame_processor into postprocessed_frames queue
        frame_processor >> postprocessed_frames;

        while (alive)
        {
            // Fetch frames from the pipeline and send them for processing
            rs2::frameset fs = pipe.wait_for_frames();
            if (fs.size() != 0) frame_processor.invoke(fs);
        }
        });

    // Skips some frames to allow for auto-exposure stabilization
    for (int i = 0; i < 30; i++) pipe.wait_for_frames();

    // Have we recorded test frames?
    bool recorded_frames = false;


    // MAIN LOOP -------------------------------------------------
    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0) // Application still alive?
    {
        static rs2::frameset frameset;
        postprocessed_frames.poll_for_frame(&frameset);

        if (frameset.size() == 0) continue;
        //Get processed aligned frame
        rs2::video_frame color_frame = frameset.get_color_frame();
        rs2::depth_frame depth_frame = frameset.get_depth_frame();

        //If one of them is unavailable, continue iteration
        if (!depth_frame || !color_frame) continue;

        float width = depth_frame.get_width();
        float height = depth_frame.get_height();


        //*
        // Remove background
        //rs2::video_frame altered_frame = color_frame;
        //remove_background(altered_frame, depth_frame, depth_scale, depth_clipping_min, depth_clipping_max);
         
        // Colorize depth image using colorizer's histogram equaliation. White=near, black = far
        color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2);
        rs2::frame bw_depth = depth_frame.apply_filter(color_map);
        // */

        
        // CV PROCESSING - SIMPLE ------------------------------------------------------
        // Create CV Mats from Color & Depth frames
        auto color_cv_frame = frame_to_mat(color_frame);
        auto depth_cv_frame = frame_to_mat(bw_depth);
        flip(color_cv_frame, color_cv_frame, 0);
        flip(depth_cv_frame, depth_cv_frame, 0);

        // Using CV BackgroundSubtractor
        Mat foreground_mask;
        background_subtractor->apply(color_cv_frame, foreground_mask, -1);

        // TRY THIS FLOODFILL technique
        // https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
        // Threshold - Set values equal to or above 220 to 0.
        // Set values below 220 to 255.
        Mat im_th;
        threshold(foreground_mask, im_th, 220, 255, THRESH_BINARY);

        // Floodfill from point (0, 0)
        Mat im_floodfill = im_th.clone();
        floodFill(im_floodfill, cv::Point(0, 0), Scalar(255));

        // Invert floodfilled image
        Mat im_floodfill_inv;
        bitwise_not(im_floodfill, im_floodfill_inv);
  
        // Combine the two images to get the foreground.
        Mat im_out = (im_th | im_floodfill_inv);

        depth_cv_mask = im_out;

        
        auto depth_cv_mask = Mat(Size(width, height), CV_8UC1);   // GRAY
        auto composite_cv_frame = Mat(Size(width, height), CV_8UC4); // RGBA
        auto display_cv_frame = Mat(Size(width, height), CV_8UC3); // RGB

        //auto depth_cv_frame = Mat(Size(width, height), CV_8UC3, (void*)bw_depth.get_data(), Mat::AUTO_STEP);
        //threshold(foreground_mask, depth_cv_mask, thresh_min_value, 255, THRESH_BINARY);
        //threshold(depth_cv_frame, depth_cv_mask, thresh_min_value, thresh_max_value, THRESH_BINARY);
        //cvtColor(depth_cv_frame, depth_cv_mask, COLOR_BGR2GRAY); // or COLOR_RGB2GRAY ?
        //create_mask_from_depth(depth_cv_frame, 180, THRESH_BINARY);

        //cvtColor(color_cv_frame, color_cv_frame, CV_BGR2RGB);

        color_cv_frame.copyTo(composite_cv_frame, depth_cv_mask); // Apply binary mask
        color_cv_frame.copyTo(display_cv_frame);
    
        int gnome_width = 192;
        int gnome_height = 108;
        int gnome_x = 50;
        int gnome_y = 50;

        auto resized_cv_frame = Mat(Size(gnome_width, gnome_height), CV_8UC4); // RGBA
        resize(composite_cv_frame,resized_cv_frame, Size(gnome_width, gnome_height));

        // Test Read Gnome Image
        char* file = "C:/Gnomes/gnome_0100.png";
        Mat gnome_image = imread(file, IMREAD_UNCHANGED);
        //cvtColor(resized_cv_frame, resized_cv_frame, CV_RGB2BGR);

        // Display Gnome with alpha on display frame
        // https://stackoverflow.com/questions/32478976/want-transparent-image-even-after-blending/32481105#32481105

        //Mat3b gnome_image = resized_cv_frame;
        //Mat3b roi = display_cv_frame(Rect(100, 50, gnome_width, gnome_height));
        //cv::addWeighted(roi, 0.5f, display_cv_frame, 1 - 0.5f, 0.0, display_cv_frame);


        // Another option using overlayImage
        //cv::cvtColor(resized_cv_frame, resized_cv_frame, CV_RGB2BGR);
        //cv::cvtColor(color_cv_frame, color_cv_frame, CV_RGB2BGR);
        overlayImage(display_cv_frame, resized_cv_frame, display_cv_frame, Point(gnome_x, gnome_y));

        // Display Frame
        //imshow("CV Frame", composite_cv_frame);
        //imshow("CV Frame", display_cv_frame);
        imshow("FG Mask", depth_cv_mask);


        // Write Test Frame Files (once)
        if (!recorded_frames)
        {
            // Calculate Time
            time_t now = time(0);
            tm* local_time = localtime(&now);
            //int year = local_time.tm_year;
            char time_char[40];  //asctime(local_time);
            strftime(time_char, 20, "%Y_%m_%d_%H:%M:%S", std::localtime(&now));
            std::string time_str (time_char);
            std::string file_path ("C:/Gnomes/"); // +time_str + "\\";

            //char* file_path_c = const_cast<char*>(file_path.c_str());
            //std::experimental::filesystem::path file_path_c = const_cast<char*>(file_path.c_str());
            //std::experimental::filesystem::create_directory(file_path_c);

            //std::string image_path = file_path + "cv_color_frame.png";

            std::ofstream log;
            log.open("log.txt");
            //log << "Writing this to a file.\n";
            log << "time = " << time_char << '\n';
            log << "file_path = " << file_path << '\n';
            log << std::string(file_path + "cv_color_" + time_str + ".png") << '\n';
            //log << "file_path_c = " << file_path_c << '\n';
            //log << "image_path = " << image_path << '\n';
            log.close();

            //CreateDirectory(file_path_c, NULL);
            //system("mkdir " + file_path);
            //imwrite(image_path, color_cv_frame);
            imwrite(file_path + "cv_color_frame.png", color_cv_frame);
            imwrite(file_path + "cv_depth_frame.png", depth_cv_frame);
            imwrite(file_path + "cv_depth_mask.png", depth_cv_mask);
            imwrite(file_path + "cv_composite_frame.png", composite_cv_frame);
            imwrite(file_path + "cv_resized_frame.png", resized_cv_frame);
            imwrite(file_path + "cv_display_frame.png", display_cv_frame);
            imwrite(file_path + "cv_foreground_mask.png", foreground_mask);
            //imwrite(std::string(file_path + "cv_color_" + time_str + ".png"), color_cv_frame);
            //imwrite(std::string(file_path + "cv_depth_" + time_str +".png"), depth_cv_frame);
            //imwrite(std::string(file_path + "cv_mask_" + time_str + ".png"), depth_cv_mask);
            //imwrite(std::string(file_path + "cv_composite_" + time_str + ".png"), composite_cv_frame);
            //imwrite(std::string(file_path + "cv_resized_" + time_str + ".png"), resized_cv_frame);
            //imwrite(std::string(file_path + "cv_display_" + time_str + ".png"), display_cv_frame);
            recorded_frames = true;
        }

        //imshow("CV Mask", depth_cv_mask);

        // SHOW GNOME ROI AS TEST
        // cv:Rect(0 ,0, resized_cv_frame,)

        // RENDER -----------------------------------------------
        /*
        // Taking dimensions of the window for rendering purposes
        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());

        // At this point, "other_frame" is an altered frame, stripped from its background
        // Calculating the position to place the frame in the window
        rect color_frame_rect{ 0, 0, w, h };
        color_frame_rect = color_frame_rect.adjust_ratio({ width, height });

        // Render aligned image
        renderer.render(color_frame, color_frame_rect);

        // Render depth frame as a picture-in-picture
        // Calculating the position to place the depth frame in the window
        rect pip_stream{ 0, 0, w / 5, h / 5 };
        pip_stream = pip_stream.adjust_ratio({ width, height });
        pip_stream.x = color_frame_rect.x + color_frame_rect.w - pip_stream.w - (std::max(w, h) / 25);
        pip_stream.y = color_frame_rect.y + (std::max(w, h) / 25);

        // Render depth (as picture in pipcture)
        renderer.upload(color_map.process(depth_frame));
        renderer.show(pip_stream);
        // */


        // Using ImGui library to provide a slide controller to select the depth clipping distance
        // Make ESC menu to show depth sliders, filter controls, etc
        //imgui_implglfw_newframe(1);
        //render_slider({ 5.f, 0, w, h }, depth_clipping_max);
        //imgui::render();

        // SAVE IMAGE TO DISK using stringstream -----------------------------------
        //*
        //std::stringstream png_file;
        //png_file << "jeffu-lilliput-test-" << color_frame.get_profile().stream_name() << ".png";
        //imwrite("..path\\lilliput-gnome-test.png", final_cv_frame);
        //stbi_write_png(png_file.str().c_str(), width, height, color_frame.get_bytes_per_pixel(), color_frame.get_data(), color_frame.get_stride_in_bytes());
        //std::cout << "Saved " << png_file.str() << std::endl;
        // */

    }
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

// CV ATTEMPTS
void simpleCV(rs2::frame color_frame, rs2::frame depth_frame, float width, float height, int thresh_value)
{
    //using namespace cv;

    // ATTEMPT to simplify Background-Removal
    //auto color_cv_frame = frame_to_mat(color_frame);
    //auto depth_cv_frame = frame_to_mat(depth_frame);
    //Mat depth_cv_mask = Mat(Size(width, height), CV_8UC1);   // GRAY
    //Mat final_cv_frame = Mat(Size(width, height), CV_8UC4); // RGBA
    //cv::cvtColor(color_cv_frame, color_cv_frame, CV_BGR2RGB);

    //cv::threshold(depth_cv_frame, depth_cv_mask, thresh_value, 255, cv::ThresholdTypes::THRESH_BINARY);

    //color_cv_frame.copyTo(final_cv_frame, depth_cv_mask); // Apply binary mask

    //char* cv_frame_window = "CV Window";
    //namedWindow(cv_frame_window, CV_WINDOW_AUTOSIZE);
    //imshow(cv_frame_window, final_cv_frame);

    //char* cv_mask_window = "CV Window";
    //namedWindow(cv_mask_window, CV_WINDOW_AUTOSIZE);
    //imshow(cv_mask_window, depth_cv_mask);

}

// Create Matted CV Image
cv::Mat3b createMattedCvImage(rs2::frame color_frame, rs2::frame depth_frame)
{
    using namespace cv;

    auto color_mat = frame_to_mat(color_frame);

    // We are using StructuringElement for erode / dilate operations
    auto gen_element = [](int erosion_size)
    {
        return getStructuringElement(MORPH_RECT,
            Size(erosion_size + 1, erosion_size + 1),
            Point(erosion_size, erosion_size));
    };

    const int erosion_size = 3;
    auto erode_less = gen_element(erosion_size);
    auto erode_more = gen_element(erosion_size * 2);

    // The following operation is taking grayscale image,
    // performs threshold on it, closes small holes and erodes the white area
    auto create_mask_from_depth = [&](Mat& depth, int thresh, ThresholdTypes type)
    {
        threshold(depth, depth, thresh, 255, type);
        dilate(depth, depth, erode_less);
        erode(depth, depth, erode_more);
    };


    // Generate "near" mask image:
    auto near = frame_to_mat(depth_frame);
    cv::cvtColor(near, near, cv::COLOR_BGR2GRAY);
    // Take just values within range [180-255]
    // These will roughly correspond to near objects due to histogram equalization
    create_mask_from_depth(near, 180, THRESH_BINARY);

    // Generate "far" mask image:
    auto far = frame_to_mat(depth_frame);
    cvtColor(far, far, COLOR_BGR2GRAY);
    far.setTo(255, far == 0); // Note: 0 value does not indicate pixel near the camera, and requires special attention 
    create_mask_from_depth(far, 100, THRESH_BINARY_INV);

    // GrabCut algorithm needs a mask with every pixel marked as either:
    // BGD, FGB, PR_BGD, PR_FGB
    Mat mask;
    mask.create(near.size(), CV_8UC1);
    mask.setTo(Scalar::all(GC_BGD)); // Set "background" as default guess
    mask.setTo(GC_PR_BGD, far == 0); // Relax this to "probably background" for pixels outside "far" region
    mask.setTo(GC_FGD, near == 255); // Set pixels within the "near" region to "foreground"

    // Run Grab-Cut algorithm:
    Mat bgModel, fgModel;
    cv::grabCut(color_mat, mask, Rect(), bgModel, fgModel, 1, GC_INIT_WITH_MASK);

    // Extract foreground pixels based on refined mask from the algorithm
    Mat3b foreground = Mat3b::zeros(color_mat.rows, color_mat.cols);
    color_mat.copyTo(foreground, (mask == GC_FGD) | (mask == GC_PR_FGD));
    return foreground;
}

// Attempt to run GrabCut algorithm (from rs-grabcuts.cpp)
cv::Mat3b runGrabCut(rs2::frame color_frame, rs2::frame depth_frame)
{
    using namespace cv;

    auto color_mat = frame_to_mat(color_frame);

    // We are using StructuringElement for erode / dilate operations
    auto gen_element = [](int erosion_size)
    {
        return getStructuringElement(MORPH_RECT,
            Size(erosion_size + 1, erosion_size + 1),
            Point(erosion_size, erosion_size));
    };

    const int erosion_size = 3;
    auto erode_less = gen_element(erosion_size);
    auto erode_more = gen_element(erosion_size * 2);

    // The following operation is taking grayscale image,
    // performs threshold on it, closes small holes and erodes the white area
    auto create_mask_from_depth = [&](Mat& depth, int thresh, ThresholdTypes type)
    {
        threshold(depth, depth, thresh, 255, type);
        dilate(depth, depth, erode_less);
        erode(depth, depth, erode_more);
    };


    // Generate "near" mask image:
    auto near = frame_to_mat(depth_frame);
    cv::cvtColor(near, near, cv::COLOR_BGR2GRAY);
    // Take just values within range [180-255]
    // These will roughly correspond to near objects due to histogram equalization
    create_mask_from_depth(near, 180, THRESH_BINARY);

    // Generate "far" mask image:
    auto far = frame_to_mat(depth_frame);
    cvtColor(far, far, COLOR_BGR2GRAY);
    far.setTo(255, far == 0); // Note: 0 value does not indicate pixel near the camera, and requires special attention 
    create_mask_from_depth(far, 100, THRESH_BINARY_INV);

    // GrabCut algorithm needs a mask with every pixel marked as either:
    // BGD, FGB, PR_BGD, PR_FGB
    Mat mask;
    mask.create(near.size(), CV_8UC1);
    mask.setTo(Scalar::all(GC_BGD)); // Set "background" as default guess
    mask.setTo(GC_PR_BGD, far == 0); // Relax this to "probably background" for pixels outside "far" region
    mask.setTo(GC_FGD, near == 255); // Set pixels within the "near" region to "foreground"

    // Run Grab-Cut algorithm:
    Mat bgModel, fgModel;
    cv::grabCut(color_mat, mask, Rect(), bgModel, fgModel, 1, GC_INIT_WITH_MASK);

    // Extract foreground pixels based on refined mask from the algorithm
    Mat3b foreground = Mat3b::zeros(color_mat.rows, color_mat.cols);
    color_mat.copyTo(foreground, (mask == GC_FGD) | (mask == GC_PR_FGD));
    return foreground;
}


// Overlay Image using CV
void overlayImage(cv::Mat &background, cv::Mat &foreground, cv::Mat &output, const Point& location)
{
    // https://answers.opencv.org/question/73016/how-to-overlay-an-png-image-with-alpha-channel-to-another-png/
    // http://jepsonsblog.blogspot.be/2012/10/overlay-transparent-image-in-opencv.html
    // https://gist.github.com/maximus5684/082f8939edb6aed7ba0a

    background.copyTo(output);

    // start at the row indicated by location, or at row 0 if location.y is negative.
    for (int y = std::max(location.y, 0); y < background.rows; ++y)
    {
        int fY = y - location.y;                // because of the translation
        if (fY >= foreground.rows) break;       // we are done if we have processed all rows of the foreground image.

        // start at the column indicated by location, or at column 0 if location.x is negative.
        for (int x = std::max(location.x, 0); x < background.cols; ++x)
        {
            int fX = x - location.x;            // because of the translation.
            if (fX >= foreground.cols) break;   // we are done with this row if the column is outside of the foreground image.

            // determine the opacity of the foreground pixel, using its fourth (alpha) channel.
            double opacity = ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.;

            // and now combine the background and foreground pixel, using the opacity, but only if opacity > 0.
            for (int c = 0; opacity > 0 && c < output.channels(); ++c)
            {
                //unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
                //unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
                //src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
                unsigned char foregroundPx =  foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c];
                output.data[y * output.step + output.channels() * x + c] = backgroundPx * (1. - opacity) + foregroundPx * opacity;
            }
        }
    }
}

// Passing both frames to remove_background so it will "strip" the background
// NOTE: in this example, we alter the buffer of the other frame, instead of copying it and altering the copy
// This behavior is not recommended in real application since the other frame could be used elsewhere
void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_min, float clipping_max)
{
    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
    uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

    int width = other_frame.get_width();
    int height = other_frame.get_height();
    int other_bpp = other_frame.get_bytes_per_pixel();

    #pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
    for (int y = 0; y < height; y++)
    {
        auto depth_pixel_index = y * width;
        for (int x = 0; x < width; x++, ++depth_pixel_index)
        {
            // Get the depth value of the current pixel
            auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];

            // Check if the depth value is invalid (<=0) or greater than the threashold
            if (pixels_distance < clipping_min || pixels_distance > clipping_max)
            {
                // Calculate the offset in other frame's buffer to current pixel
                auto offset = depth_pixel_index * other_bpp;

                // Set pixel to "background" color (0x999999)
                // WAIT -- How do I mask to alpha?
                std::memset(&p_other_frame[offset], 0x000000, other_bpp);
            }
        }
    }
}

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}

// Render Slider
/*
void render_slider(rect location, float& clipping_dist)
{
    // Some trickery to display the control nicely
    static const int flags = ImGuiWindowFlags_NoCollapse
        | ImGuiWindowFlags_NoScrollbar
        | ImGuiWindowFlags_NoSavedSettings
        | ImGuiWindowFlags_NoTitleBar
        | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoMove;
    const int pixels_to_buttom_of_stream_text = 25;
    const float slider_window_width = 30;

    ImGui::SetNextWindowPos({ location.x, location.y + pixels_to_buttom_of_stream_text });
    ImGui::SetNextWindowSize({ slider_window_width + 20, location.h - (pixels_to_buttom_of_stream_text * 2) });

    //Render the vertical slider
    ImGui::Begin("slider", nullptr, flags);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImColor(215.f / 255, 215.0f / 255, 215.0f / 255));
    auto slider_size = ImVec2(slider_window_width / 2, location.h - (pixels_to_buttom_of_stream_text * 2) - 20);
    ImGui::VSliderFloat("", slider_size, &clipping_dist, 0.0f, 6.0f, "", 1.0f, true);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Depth Clipping Max: %.3f", clipping_dist);
    ImGui::PopStyleColor(3);

    //Display bars next to slider
    float bars_dist = (slider_size.y / 6.0f);
    for (int i = 0; i <= 6; i++)
    {
        ImGui::SetCursorPos({ slider_size.x, i * bars_dist });
        std::string bar_text = "- " + std::to_string(6-i) + "m";
        ImGui::Text("%s", bar_text.c_str());
    }
    ImGui::End();
}
*/

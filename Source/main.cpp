//
//  main.cpp
//  facerecognizer
//
//  Created by Jimesh Thomas on 02/10/16.
//  Copyright © 2016 JT. All rights reserved.
//  Code referred from http://docs.opencv.org/
//

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>


#define CAPTURED_IMAGE      "Captured_Image.jpg"
#define WINDOW_TITLE        "Faces Detected"
#define HAAR_CASCADE_XML    "haarcascade_frontalface_default.xml"


using namespace cv;
using namespace std;

//Convert the time stamp string in the format mm:ss:sss to double milli second value
double TimeStampValue(string &timeStamp)
{
    stringstream ss;
    ss.str(timeStamp);
    string item;
    char delim = ':';
    double timeStampValue;
    const size_t times = 3;
    for (size_t i = 0; i < times; i++)
    {
        getline(ss, item, delim);
        if (!item.empty())
        {
            int unitTime = 0;
            try
            {
                unitTime = std::stoi(item);

            } catch (...)
            {
                cerr << "Time data cannot be detected. Please try again!" << endl;
                exit(1);
            }
            switch (i) {
                case 0:
                    timeStampValue += unitTime * 60;
                    break;
                case 1:
                    timeStampValue += unitTime;
                    break;
                case 2:
                    timeStampValue *= 1000;
                    timeStampValue += unitTime;
                    break;
                    
                default:
                    break;
            }
        }
    }
    return timeStampValue;
}

int main(int argc, const char * argv[])
{
    string fileName;
    string timeStampString;

    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc == 3)
    {
        fileName = string(argv[1]);
        timeStampString = string(argv[2]);
    }
    else
    {
        cout << "usage: " << argv[0] << " <video file name>  <time stamp(mm:ss:sss)> " << endl;
        exit(1);
    }

    string fn_haar = string(HAAR_CASCADE_XML);

    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    
    VideoCapture video = VideoCapture(fileName);
    
    double frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);
    double fps = video.get(CV_CAP_PROP_FPS);
    
    double totalDuration = (frameCount/fps) * 1000;
    
    double timeStamp = TimeStampValue(timeStampString);
    
    if (timeStamp > totalDuration)
    {
        cerr << "Selected time is greater than total duration of the video. Please try again!" << endl;
        exit(1);
    }
    
    video.set(CV_CAP_PROP_POS_MSEC,timeStamp);
    
    Mat original;
    if (video.read(original))
    {
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        // At this point you have the position of the faces in faces.
        for (int i = 0; i < faces.size(); i++)
        {
            // Process face by face:
            cv::Rect face_i = faces[i];
            // Crop the face from the image and save to file.
            Mat face = gray(face_i);
            string imageName = format("Face_Image_%d.jpg", i);
            imwrite( imageName, face);
            imshow(imageName, face);
            // Draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
        }
        
        //Annotate the captured frame with time stamp:
        string frame_text = "Time Stamp = " + timeStampString;
        putText(original, frame_text, cv::Point(20, 10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

        imwrite(CAPTURED_IMAGE, original);

        // Show the result:
        imshow(WINDOW_TITLE, original);
        
        int k = waitKey(0);
        if (k == 27)         //wait for ESC key to exit
        {
            cv::destroyAllWindows();
        }
    }
    else
    {
        printf( "No video data \n" );
        return -1;
    }
    return 0;
}

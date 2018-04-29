/*
 *  Author: Divya Sampath Kumar, Harsimransingh Bindra
 *	Description: The code calculates average time taken to compute Hough 
				 Elliptical transform of 100 frames of an image. clock_gettime() is
				 used to calculate the time. The frame sizes under consid-
				 eration are: 320*240, 640*480, 1280*960. 
 *  Example by Sam Siewert
 *
 */
 
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;



char timg_window_name_houghline_eliptical[] = "Houghline Elliptical Transform";
CvCapture* capture;
Mat copied;

/* Macros */
#define NSEC_PER_SEC (1000000000)
#define NSEC_PER_MSEC (1000000)
#define NSEC_PER_MICROSEC (1000)


pthread_t detectThread;
pthread_attr_t main_sched_attr, thread_attr;

/* Function definition for delta_t */
int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec=stop->tv_sec - start->tv_sec;
  int dt_nsec=stop->tv_nsec - start->tv_nsec;

  if(dt_sec >= 0)
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }
  else
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }
  return 1;
}



void* BallDetection(void *arg)
{
   static struct timespec start_time = {0, 0};
   static struct timespec stop_time = {0, 0};
   static struct timespec final_time = {0, 0};
   CvCapture* capture;
   IplImage* frame;
   Mat gray;
   vector<Vec3f> circles;
   cvNamedWindow(timg_window_name_houghline_eliptical, CV_WINDOW_AUTOSIZE);
   
   capture = (CvCapture *)cvCreateCameraCapture(0);
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
	
   //Start Timer
   clock_gettime(CLOCK_REALTIME, &start_time);
   int numberOfFrames = 0;

//crop image
   /*int offset_x = 129;
   int offset_y = 129;
   Rect roi;
   roi.x = offset_x;
   roi.y = offset_y;*/
   int width = 160;
   int height = 120;
   int x, y = 0;
   
   
   while(numberOfFrames <1000)
   {
      
      frame=cvQueryFrame(capture);

      Mat mat_frame(frame);
     // Mat gray(frame);
      
      cvtColor(mat_frame, gray, CV_BGR2GRAY);
      GaussianBlur(gray, gray, Size(11,15), 2, 2);
    
      Rect rect(x,y,width,height);
      Point pt1(80,60);
      Point pt2(240,180);
//crop image
      roi.width = gray.size().width - (offset_x);
      roi.height = gray.size().height - (offset_y);
      Mat crop = gray(roi);
      copied = gray.clone();
      HoughCircles(copied, circles, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 50, 0, 0);
      for(size_t i = 0; i < circles.size(); i++ )
      {
    	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    	int radius = cvRound(circles[i][2]);
    	// circle center
    	circle( mat_frame, center, 1, Scalar(0,255,0), -1, 8, 0 );
    	// circle outline
        circle( mat_frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
      }
      rectangle(copied,pt1,pt2,Scalar(0,255,0),2,8,0);
      if(!frame) break;

      cvShowImage(timg_window_name_houghline_eliptical, frame);
   //     imwrite("Cropped.png", crop);
      imshow("Rectangle image", copied);
      char c = cvWaitKey(10);
      if( c == 27 ) break;
      numberOfFrames++;     
    }
    
    //Stop timer
    clock_gettime(CLOCK_REALTIME, &stop_time);

    delta_t(&stop_time,&start_time,&final_time);
    printf("\nRun Time for %d frames:%ld sec %ldmsec\n",numberOfFrames,final_time.tv_sec,	(final_time.tv_nsec)/NSEC_PER_MSEC);

    cvReleaseCapture(&capture);
    cvDestroyWindow(timg_window_name_houghline_eliptical);
}

int main( int argc, char** argv )
{
    pthread_attr_init(&thread_attr);
    if (pthread_create(&detectThread, &thread_attr,BallDetection, NULL) != 0)
    {
        printf("Failed to create ball detect thread\n");
    } 
    pthread_join(detectThread,NULL);
    
};

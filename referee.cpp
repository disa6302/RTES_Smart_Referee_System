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
#include <signal.h>
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
pthread_mutex_t mutex_locker =PTHREAD_MUTEX_INITIALIZER;

volatile int lookup[3]={0};
char timg_window_name_houghline_eliptical[] = "Houghline Elliptical Transform";
CvCapture* capture;
Mat copied,copied2;
volatile sig_atomic_t detect1, detect2;

/* Macros */
#define NSEC_PER_SEC (1000000000)
#define NSEC_PER_MSEC (1000000)
#define NSEC_PER_MICROSEC (1000)

#define GOALS 1000
#define ROI2_x_offset 40
#define ROI2_y_offset 30
#define ROI1_x_offset 80
#define ROI1_y_offset 60

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

void HoughCalc_ROIcheck(Mat &gray,int ROI1_x,int ROI1_y,int ROI2_x,int ROI2_y)
{
      vector<Vec3f> circles1,circles2;
      Rect roi1,roi2;
      
      roi2.width = gray.size().width - (ROI2_x*2);
      roi2.height = gray.size().height - (ROI2_y*2);
      Mat crop_ROI2 = gray(roi2);
      Mat crop_ROI1;
      cvtColor(crop_ROI2, copied2, CV_BGR2GRAY);

      roi1.width = crop_ROI2.size().width - ((18)*2);
      roi1.height = crop_ROI2.size().height - ((13.5)*2);
      crop_ROI1 = crop_ROI2(roi1);
      cvtColor(crop_ROI1, copied, CV_BGR2GRAY);

      int detect_type =0;
      /*if(ROI_x== ROI2_x_offset)
      	  detect_type = 2;
      if(ROI_x== ROI1_x_offset)
      	  detect_type = 1;*/       


      HoughCircles(copied2, circles2, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 36, 0, 0);
      if(circles2.size()>0) 
	{
		pthread_mutex_lock(&mutex_locker);
		detect2 = 1;
		pthread_mutex_unlock(&mutex_locker); 


	}
	HoughCircles(copied, circles1, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 36, 0, 0);
      	if(circles1.size()>0) 
	{
		pthread_mutex_lock(&mutex_locker);
		detect1 = 1;
		pthread_mutex_unlock(&mutex_locker); 
	}
		
      for(size_t i = 0; i < circles1.size(); i++ )
      {
    
	
    	Point center1(cvRound(circles1[i][0]), cvRound(circles1[i][1]));
    	int radius1 = cvRound(circles1[i][2]);

    	//circle( mat_frame, center, 1, Scalar(0,255,0), -1, 8, 0 );
	circle( crop_ROI1, center1, 1, Scalar(0,255,0), -1, 8, 0 );
    	// circle outline
	// circle(mat_frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
        circle( crop_ROI1, center1, radius1, Scalar(0,0,255), 3, 8, 0 );
      } 

      for(size_t i = 0; i < circles2.size(); i++ )
      {
    
	
    	Point center2(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
    	int radius2 = cvRound(circles2[i][2]);

    	//circle( mat_frame, center, 1, Scalar(0,255,0), -1, 8, 0 );
	circle( crop_ROI2, center2, 1, Scalar(0,255,0), -1, 8, 0 );
    	// circle outline
	// circle(mat_frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
        circle( crop_ROI2, center2, radius2, Scalar(0,0,255), 3, 8, 0 );
      } 
     	
	imshow("ROI2_Image", crop_ROI2);
	imshow("ROI1_Image", crop_ROI1);


 
}

void HoughCalc_ROI2(Mat &gray,int ROI_x,int ROI_y)
{
      vector<Vec3f> circles;
      Rect roi;
      roi.width = gray.size().width - (ROI_x*2);
      roi.height = gray.size().height - (ROI_y*2);
      Mat crop_ROI = gray(roi);
      cvtColor(crop_ROI, copied, CV_BGR2GRAY);
      int detect_type =0;
      /*if(ROI_x== ROI2_x_offset)
      	  detect_type = 2;
      if(ROI_x== ROI1_x_offset)
      	  detect_type = 1;*/       

      HoughCircles(copied, circles, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 36, 0, 0);
      if(circles.size()>0) 
	{
		pthread_mutex_lock(&mutex_locker);
		detect2 = 1;
		pthread_mutex_unlock(&mutex_locker); 
	}
      for(size_t i = 0; i < circles.size(); i++ )
      {
      		
    	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    	int radius = cvRound(circles[i][2]);

    	//circle( mat_frame, center, 1, Scalar(0,255,0), -1, 8, 0 );
	circle( crop_ROI, center, 1, Scalar(0,255,0), -1, 8, 0 );
    	// circle outline
	// circle(mat_frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
        circle( crop_ROI, center, radius, Scalar(0,0,255), 3, 8, 0 );
      }
 
     	imshow("ROI2_Image", crop_ROI);
 
}

void* BallDetection(void *arg)
{


   static struct timespec start_time = {0, 0};
   static struct timespec stop_time = {0, 0};
   static struct timespec final_time = {0, 0};
   CvCapture* capture;
   IplImage* frame;
   Mat gray;
   //vector<Vec3f> circles1,circles2;
   cvNamedWindow(timg_window_name_houghline_eliptical, CV_WINDOW_AUTOSIZE);
   
   capture = (CvCapture *)cvCreateCameraCapture(0);
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
	
   //Start Timer
   clock_gettime(CLOCK_REALTIME, &start_time);
   int numberOfFrames = 0;

//crop image
//50%

   

//25%
/*
   int ROI1_x = 120;
   int ROI1_y = 90;
*/
//75% crop

/*
(0,0)
---------------------------
   (40,30)     (40,210)  
    ------------
    |          |
    |          |
    ------------
   (280,30)    (280,210)
---------------------------
 			  (320,240)	
*/

  int ROI2_x = ROI2_x_offset;
  int ROI2_y = ROI2_y_offset;

//50% 

/*
(0,0)
---------------------------
   (80,60)     (80,180)  
    ------------
    |          |
    |          |
    ------------
   (240,60)    (240,180)
---------------------------
 			  (320,240)	
*/
int ROI1_x = ROI1_x_offset;
  int ROI1_y = ROI1_y_offset;

//25% 

/*


(0,0)
---------------------------
   (120,90)     (120,150)  
    ------------
    |          |
    |          |
    ------------
   (200,90)    (200,150)
---------------------------
 			  (320,240)


*/


	
   Rect roi1, roi2;
   roi1.x = ROI1_x;
   roi1.y = ROI1_y;
   roi2.x = ROI2_x;
   roi2.y = ROI2_x;
   int width = 160;
   int height = 120;
   int x, y = 0;
   
//ROI2 = outer, ROI1 = inner
   
   while(numberOfFrames <GOALS)
   {
      
      frame=cvQueryFrame(capture);

      Mat mat_frame(frame);
     // Mat gray(frame);
     gray = mat_frame.clone();
     // cvtColor(mat_frame, gray, CV_BGR2GRAY);
     // GaussianBlur(gray, gray, Size(3,3), 2, 2);

	//crop image
      HoughCalc_ROIcheck(gray,ROI1_x,ROI1_y,ROI2_x,ROI2_y);
      //HoughCalc_ROI1(gray,ROI1_x,ROI1_y);
      //HoughCalc_ROI2(gray,ROI2_x,ROI2_y);
           


      //rectangle(copied,pt1,pt2,Scalar(0,255,0),2,8,0);
      if(!frame) break;

      cvShowImage(timg_window_name_houghline_eliptical, frame);

      
      printf("[Frame Num :%d] detect1:%d,detect2:%d\n",numberOfFrames,detect1,detect2);
      /*if(detect1 == 1)
      {
	//printf("Goal\n");
	lookup[0]++;
	detect1 = 0;
      }
      else
      {
	lookup[2]++;
      }*/
      pthread_mutex_lock(&mutex_locker); 
      if(!detect1 && !detect2)
      {
	lookup[2]++;
 	//printf("Invalid throw\n");
      }
      else if(detect1 && detect2)
      {
	lookup[0]++;
	//printf("Goal\n");
	detect1 = detect2 = 0 ;
      }
      else if(detect2 && !detect1) 
      {
	//printf("No Goal!\n");
	lookup[1]++;
	detect2 = 0;
      }
      pthread_mutex_unlock(&mutex_locker); 
      char c = cvWaitKey(10);
      if( c == 27 ) break;
      numberOfFrames++;     
    }
    
    //Stop timer
    clock_gettime(CLOCK_REALTIME, &stop_time);

    delta_t(&stop_time,&start_time,&final_time);
    printf("\nRun Time for %d frames:%ld sec %ldmsec\n",numberOfFrames,final_time.tv_sec,(final_time.tv_nsec)/NSEC_PER_MSEC);

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
    printf("Goals:%d,No Goals:%d,Invalid:%d,Total:%d\n",lookup[0],lookup[1],lookup[2],(lookup[0]+lookup[1]+lookup[2]));
    
};

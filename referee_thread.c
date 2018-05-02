/*
 *  Author: Divya Sampath Kumar, Bhallaji Venkatesan
 
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
pthread_mutex_t mutex_locker = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_locker = PTHREAD_COND_INITIALIZER;

volatile int lookup[3]={0};
char timg_window_name_houghline_eliptical[] = "Houghline Elliptical Transform";
Mat crop_ROI2,crop_ROI1;

Mat copied,copied2;
volatile sig_atomic_t detect1, detect2, signal_capture = 0;

/* Macros */
#define NSEC_PER_SEC (1000000000)
#define NSEC_PER_MSEC (1000000)
#define NSEC_PER_MICROSEC (1000)

#define GOALS 100
#define ROI2_x_offset 40
#define ROI2_y_offset 30
#define ROI1_x_offset 80
#define ROI1_y_offset 60

pthread_t detectThread,captureThread;
pthread_attr_t main_sched_attr, thread_attr;

CvCapture* capture;
IplImage* frame;
Mat gray;

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

static struct timespec start1_time = {0, 0};
    static struct timespec stop1_time = {0, 0};
    static struct timespec final1_time = {0, 0};

void* BallDetection(void *arg)
{
    
    vector<Vec3f> circles1,circles2;
    Rect roi1,roi2;
    int numberOfFrames = GOALS;
    clock_gettime(CLOCK_REALTIME, &start1_time);
    while(numberOfFrames > 0)
    {
	    pthread_mutex_lock(&mutex_locker);
	    pthread_cond_wait(&cond_locker,&mutex_locker);
	    //printf("got the lock 1st in detect%d\n",numberOfFrames);


	    roi2.width = gray.size().width - (ROI2_x_offset*2);
	    roi2.height = gray.size().height - (ROI2_y_offset*2);
	    crop_ROI2 = gray(roi2);
	   
	    cvtColor(crop_ROI2, copied2, CV_BGR2GRAY);

	    roi1.width = crop_ROI2.size().width - ((18)*2);
	    roi1.height = crop_ROI2.size().height - ((13.5)*2);
	    crop_ROI1 = crop_ROI2(roi1);

	    cvtColor(crop_ROI1, copied, CV_BGR2GRAY);       

	
	    HoughCircles(copied2, circles2, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 36, 0, 0);
	    if(circles2.size()>0) 
	    {
		detect2 = 1;
	    }
	
	    HoughCircles(copied, circles1, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 36, 0, 0);
	    if(circles1.size()>0) 
	    {
		detect1 = 1; 
	    }
		
	    for(size_t i = 0; i < circles1.size(); i++ )
	    {
	    	Point center1(cvRound(circles1[i][0]), cvRound(circles1[i][1]));
	    	int radius1 = cvRound(circles1[i][2]);
		circle(crop_ROI1, center1, 1, Scalar(0,255,0), -1, 8, 0 );
		circle(crop_ROI1, center1, radius1, Scalar(0,0,255), 3, 8, 0 );
	    } 

	    for(size_t i = 0; i < circles2.size(); i++ )
	    {
	    	Point center2(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
	    	int radius2 = cvRound(circles2[i][2]);
		circle(crop_ROI2, center2, 1, Scalar(0,255,0), -1, 8, 0 );
		circle(crop_ROI2, center2, radius2, Scalar(0,0,255), 3, 8, 0 );
	    }  	
       	    
	    	   
	    if(!detect1 && !detect2)
	    {
		lookup[2]++;
	    }
	    else if(detect1 && detect2)
	    {
		lookup[0]++;
	  	detect1 = detect2 = 0 ;
	    }
	    else if(detect2 && !detect1) 
            {
		lookup[1]++;
		detect2 = 0;
	    }
	    //printf("[Frame Num :%d] detect1:%d,detect2:%d\n",numberOfFrames,detect1,detect2);
	    numberOfFrames--;
	    pthread_mutex_unlock(&mutex_locker);
	    printf("Reached here %d\n",numberOfFrames);
	    signal_capture = 1;
	    
	}//end of while
	
	clock_gettime(CLOCK_REALTIME, &stop1_time);
	delta_t(&stop1_time,&start1_time,&final1_time);
    	printf("\n[Ball Detection]Run Time for %d frames:%ld sec %ldmsec\n",GOALS,final1_time.tv_sec,(final1_time.tv_nsec)/NSEC_PER_MSEC);
	pthread_exit(NULL);
}

void* BallCapture(void *arg)
{
    static struct timespec start_time = {0, 0};
    static struct timespec stop_time = {0, 0};
    static struct timespec final_time = {0, 0};
    cvNamedWindow(timg_window_name_houghline_eliptical, CV_WINDOW_AUTOSIZE);
   
    capture = (CvCapture *)cvCreateCameraCapture(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
	
    //Start Timer
    clock_gettime(CLOCK_REALTIME, &start_time);
    int numberOfFrames = GOALS;


	/*75%
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

//

	/*50% 
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



	/*25% 

		int ROI1_x = 120;
		int ROI1_y = 90;


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


   
   //ROI2 = outer, ROI1 = inner
   
   while(numberOfFrames > 0)
   {
        pthread_mutex_lock(&mutex_locker);
	//printf("got the lock 1st in capture %d\n",numberOfFrames);
	frame=cvQueryFrame(capture);

	Mat mat_frame(frame);
	//gray = mat_frame;
	gray = mat_frame.clone();
	numberOfFrames--;  
	pthread_cond_signal(&cond_locker);
	pthread_mutex_unlock(&mutex_locker);
        if(!frame) break;
        cvShowImage(timg_window_name_houghline_eliptical, frame);      
        imshow("ROI2_Image", crop_ROI2);
	imshow("ROI1_Image", crop_ROI1);
	char c = cvWaitKey(5);
	if( c == 27 ) break;
	//usleep(100000); //0.1 second delay to synchronize both threads
	while(signal_capture==0);
	signal_capture = 0;
   }
    
    //Stop timer
    clock_gettime(CLOCK_REALTIME, &stop_time);

    delta_t(&stop_time,&start_time,&final_time);
    printf("\n[Ball Capture]Run Time for %d frames:%ld sec %ldmsec\n",GOALS,final_time.tv_sec,(final_time.tv_nsec)/NSEC_PER_MSEC);
    
    cvReleaseCapture(&capture);
    cvDestroyWindow(timg_window_name_houghline_eliptical);
    pthread_exit(NULL);
}


int main( int argc, char** argv )
{
    pthread_attr_init(&thread_attr);
    if (pthread_create(&captureThread, &thread_attr,BallCapture, NULL) != 0)
    {
        printf("Failed to create ball detect thread\n");
    } 
    if (pthread_create(&detectThread, &thread_attr,BallDetection, NULL) != 0)
    {
        printf("Failed to create ball detect thread\n");
    }
    
    pthread_join(captureThread,NULL);
    pthread_join(detectThread,NULL);
    printf("Goals:%d,No Goals:%d,Invalid:%d,Total:%d\n",lookup[0],lookup[1],lookup[2],(lookup[0]+lookup[1]+lookup[2]));
    
};

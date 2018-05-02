/*
 *  Author: Divya Sampath Kumar, Bhallaji Venkatesan
 
 */


#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <signal.h>
#include <pthread.h>


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>

#include <syslog.h>
#include <sys/time.h>

#include <errno.h>

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
#define NUM_CPU_CORES (1)


#define GOALS 100
#define ROI2_x_offset 40
#define ROI2_y_offset 30
#define ROI1_x_offset 80
#define ROI1_y_offset 60

pthread_t detectThread,captureThread;
pthread_attr_t main_sched_attr, capture_attr, detect_attr, speaker_attr;

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



void* BallDetection(void *arg)
{
    static struct timespec start_time = {0, 0};
    static struct timespec stop_time = {0, 0};
    static struct timespec final_time = {0, 0};
    vector<Vec3f> circles1,circles2;
    Rect roi1,roi2;
    unsigned long int avgWCET=0, maxWCET=0;
    int numberOfFrames = GOALS;

    while(numberOfFrames > 0)
    {

	    pthread_mutex_lock(&mutex_locker);
	    pthread_cond_wait(&cond_locker,&mutex_locker);
	    
	    clock_gettime(CLOCK_REALTIME, &start_time);
	   
	
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

	    numberOfFrames--;
	    
	    imshow("ROI2_Image", crop_ROI2);
	    imshow("ROI1_Image", crop_ROI1);
	    clock_gettime(CLOCK_REALTIME, &stop_time);
	    delta_t(&stop_time,&start_time,&final_time);
	    pthread_mutex_unlock(&mutex_locker);
	    signal_capture = 1;
	    
	  
	    avgWCET += (final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000;
	    if(maxWCET < ((final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000))
	    	maxWCET = (final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000;
	
	    start_time.tv_sec = start_time.tv_nsec  = 0;
    	    stop_time.tv_sec = stop_time.tv_nsec = 0; 
    	    final_time.tv_sec = final_time.tv_nsec = 0;
	}//end of while
	
	printf("\n[Ball Detection]ACET over %d frames:%ldmsec\n",GOALS,avgWCET/GOALS);
	printf("\n[Ball Detection]WCET over %d frames:%ldmsec\n",GOALS,maxWCET);
	pthread_exit(NULL);
}

void* BallCapture(void *arg)
{
    static struct timespec start_time = {0, 0};
    static struct timespec stop_time = {0, 0};
    static struct timespec final_time = {0, 0};
    unsigned long avgWCET, maxWCET = 0;
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
	clock_gettime(CLOCK_REALTIME, &start_time);

	frame=cvQueryFrame(capture);

	Mat mat_frame(frame);
	gray = mat_frame.clone();
	numberOfFrames--;  
	
	pthread_cond_signal(&cond_locker);
	pthread_mutex_unlock(&mutex_locker);
        if(!frame) break;
        cvShowImage(timg_window_name_houghline_eliptical, frame);      
	char c = cvWaitKey(5);
	if( c == 27 ) break;
	clock_gettime(CLOCK_REALTIME, &stop_time);

   	delta_t(&stop_time,&start_time,&final_time);
	avgWCET += (final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000;
	if(maxWCET < ((final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000))
	    	maxWCET = (final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000;
	start_time.tv_sec = start_time.tv_nsec  = 0;
    	stop_time.tv_sec = stop_time.tv_nsec = 0; 
    	final_time.tv_sec = final_time.tv_nsec = 0;
	while(signal_capture==0);
	signal_capture = 0;
   }
    
   
    printf("\n[Ball Capture]ACET over %d frames:%ldmsec\n",GOALS,avgWCET/GOALS);
    printf("\n[Ball Capture]WCET over %d frames:%ldmsec\n",GOALS,maxWCET);
    cvReleaseCapture(&capture);
    cvDestroyWindow(timg_window_name_houghline_eliptical);
    pthread_exit(NULL);
}

void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
       case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
       case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n"); exit(-1);
         break;
       case SCHED_RR:
           printf("Pthread Policy is SCHED_RR\n"); exit(-1);
           break;
       default:
           printf("Pthread Policy is UNKNOWN\n"); exit(-1);
   }
}

int main( int argc, char** argv )
{
    pthread_attr_init(&capture_attr);
    pthread_attr_init(&detect_attr); 
    pthread_attr_init(&speaker_attr);

    int i, rc, scope;
    cpu_set_t threadcpu;
    int rt_max_prio, rt_min_prio;
    struct sched_param capture_prio,detect_prio, speaker_prio;
    struct sched_param main_param;
    pthread_attr_t main_attr;
    pid_t mainpid;
    cpu_set_t allcpuset;
    struct timeval current_time_val,start_time_val;
    printf("Starting Referee Demo\n");
    gettimeofday(&start_time_val, (struct timezone *)0);
    gettimeofday(&current_time_val, (struct timezone *)0);
   

    CPU_ZERO(&allcpuset);

    for(i=0; i < NUM_CPU_CORES; i++)
       CPU_SET(i, &allcpuset);

    printf("Using CPUS=%d from total available.\n", CPU_COUNT(&allcpuset));

    mainpid=getpid();

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    rc=sched_getparam(mainpid, &main_param);
    main_param.sched_priority=rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0) perror("main_param");
    print_scheduler();
  
    
    pthread_attr_getscope(&main_attr, &scope);

    if(scope == PTHREAD_SCOPE_SYSTEM)
      printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
      printf("PTHREAD SCOPE PROCESS\n");
    else
      printf("PTHREAD SCOPE UNKNOWN\n");

    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);

    CPU_ZERO(&threadcpu);
    CPU_SET(3, &threadcpu);

    rc=pthread_attr_setinheritsched(&capture_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&capture_attr, SCHED_FIFO);

    rc=pthread_attr_setinheritsched(&detect_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&detect_attr, SCHED_FIFO);

    rc=pthread_attr_setinheritsched(&speaker_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&speaker_attr, SCHED_FIFO);

    capture_prio.sched_priority=rt_max_prio-1;
    pthread_attr_setschedparam(&capture_attr, &capture_prio);

    detect_prio.sched_priority=rt_max_prio-2;
    pthread_attr_setschedparam(&detect_attr, &detect_prio);


    speaker_prio.sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&speaker_attr, &speaker_prio);

    
    printf("Threads will run on %d CPU cores\n", CPU_COUNT(&threadcpu));
    printf("Thread Priorities\n[Ball Capture]		%d\n[Ball Detection]	%d\n[Speaker]		%d\n",capture_prio.sched_priority,detect_prio.sched_priority,speaker_prio.sched_priority);
   
    
    if (pthread_create(&captureThread, &capture_attr,BallCapture, NULL) != 0)
    {
        printf("Failed to create ball detect thread\n");
    } 
    if (pthread_create(&detectThread, &detect_attr,BallDetection, NULL) != 0)
    {
        printf("Failed to create ball detect thread\n");
    }
    
    pthread_join(captureThread,NULL);
    pthread_join(detectThread,NULL);
    printf("Goals:%d,No Goals:%d,Invalid:%d,Total:%d\n",lookup[0],lookup[1],lookup[2],(lookup[0]+lookup[1]+lookup[2]));
    
}

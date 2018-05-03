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

#include <string.h>
	#include <malloc.h>
	#include <espeak/speak_lib.h>

#define WAITING   0
#define GOAL      1
#define NOGOAL    2
#define INVALID   3
#define WAIT      4
#define CONTINUE  5
#define ENDOFGAME 6
#define START 	  7	
#define PROCESSING 8

using namespace cv;
using namespace std;
pthread_mutex_t mutex_locker = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_locker = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_locker_results = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_locker_results = PTHREAD_COND_INITIALIZER;
sem_t sem1,sem2,sem3;

volatile int lookup[3]={0};
char timg_window_name_houghline_eliptical[] = "Houghline Elliptical Transform";
Mat crop_ROI2,crop_ROI1;

Mat copied,copied2;
volatile sig_atomic_t detect1, detect2, signal_capture = 0;
volatile sig_atomic_t goal_status = WAITING;

/* Macros */
#define NSEC_PER_SEC (1000000000)
#define NSEC_PER_MSEC (1000000)
#define NSEC_PER_MICROSEC (1000)
#define NUM_CPU_CORES (1)


#define GOALS 200
#define ROI2_x_offset 40
#define ROI2_y_offset 30
#define ROI1_x_offset 80
#define ROI1_y_offset 60
#define TRIALS 5


pthread_t detectThread,captureThread,resultsThread;
pthread_attr_t main_sched_attr, capture_attr, detect_attr,results_attr;

CvCapture* capture;
IplImage* frame;
Mat gray;
volatile sig_atomic_t numchances = TRIALS, goalcount = 0,game_state = START;

//enum goalStatus {GOAL,NOGOAL,INVALID}status_t;


espeak_POSITION_TYPE position_type;
espeak_AUDIO_OUTPUT output;
char *path=NULL;
int Buflength = 500, Options=0;
void* user_data;
t_espeak_callback *SynthCallback;
espeak_PARAMETER Parm;



char text_goal[30] = {"Goal....Goal....Goal"};
char text_nogoal[30] = {"No Goal"};
char text_invalid[30] = {"Invalid throw"};
char text_eog[30] = {"End of Game!"};
char text_continue[30] = {"Continue Game!"};
unsigned int size_speak,position=0, end_position=0, flags=espeakCHARS_AUTO, *unique_identifier;

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


void* ProcessResults(void* arg)
{

  while(numchances!=0)
  {
    //pthread_cond_wait(&cond_locker_results,&mutex_locker_results);

    sem_wait(&sem3);
    //while(goal_status == WAITING);
    printf("\n Goal status Received");
    game_state = WAIT;
    
    if (goal_status == GOAL)
    {
	goalcount++;
	//lookup[GOAL-1]++;
        numchances--; 	
 	printf("\n:Goal\n");
	size_speak = strlen(text_goal)+1;
        printf("Saying '%s'",text_goal);
        espeak_Synth( text_goal, size_speak, position, position_type, end_position, flags,
        unique_identifier, user_data );
        espeak_Synchronize( );
        printf("\n:Done\n"); 
    }
    else if (goal_status == NOGOAL)
    {
	//lookup[NOGOAL-1]++;
	numchances--;
	printf("\n NO Goal\n");
	size_speak = strlen(text_nogoal)+1;
        printf("Saying '%s'",text_nogoal);
        espeak_Synth(text_nogoal, size_speak, position, position_type, end_position, flags,
        unique_identifier, user_data );
        espeak_Synchronize( );
        printf("\n:Done\n");
    }
    else if (goal_status == INVALID)
    {
	//lookup[INVALID-1]++;
	numchances--;
  	printf("\n:Invalid\n");
	size_speak = strlen(text_invalid)+1;
        printf("Saying '%s'",text_invalid);
        espeak_Synth( text_invalid, size_speak, position, position_type, end_position, flags,
        unique_identifier, user_data );
        espeak_Synchronize( );
        printf("\n:Done\n");  
    }
    goal_status = WAITING;
    if(!numchances)
    {
	char text_goalstats[30];
        sprintf(text_goalstats,"Number of Goals Scored is %d. End of Game!",goalcount);
        size_speak = strlen(text_goalstats)+1;
    	printf("Saying '%s'",text_goalstats);
    	espeak_Synth( text_goalstats, size_speak, position, position_type, end_position, flags,
    	unique_identifier, user_data );
    	espeak_Synchronize( );
    	printf("\n:Done\n");
	game_state = ENDOFGAME;
	sem_post(&sem1);
	sem_post(&sem2);
	//speak out end of game
    }
    else
    { 
        char text_goalstats[30];
        sprintf(text_goalstats,"Number of Goals Scored is %d, Continue Game!",goalcount);
        size_speak = strlen(text_goalstats)+1;
    	printf("Saying '%s'",text_goalstats);
    	espeak_Synth( text_goalstats, size_speak, position, position_type, end_position, flags,
    	unique_identifier, user_data );
    	espeak_Synchronize( );
    	printf("\n:Done\n");
        game_state = CONTINUE;
	printf("number of chances:%d\n",numchances);
	sem_post(&sem1);
	sem_post(&sem2);
	
    }  

  }

    pthread_exit(NULL);
     	
       

}

void* BallDetection(void *arg)
{
    /*cpu_set_t thread2_cpuset;
    CPU_ZERO(&thread2_cpuset);
    CPU_SET(1, &thread2_cpuset);
    sched_setaffinity(0,sizeof(thread2_cpuset),&thread2_cpuset);*/
    static struct timespec start_time = {0, 0};
    static struct timespec stop_time = {0, 0};
    static struct timespec final_time = {0, 0};
    vector<Vec3f> circles1,circles2;
    Rect roi1,roi2;
    unsigned long int avgWCET=0, maxWCET=0;
    int numberOfFrames = GOALS;
    while(numchances >0)
    {

   
    sem_wait(&sem2);
    if(game_state == ENDOFGAME)
	break;
    //while((game_state == START) || (game_state == CONTINUE))//game_state!=WAIT && game_state!=ENDOFGAME && game_state!=PROCESSING)
    //while(game_state!=WAIT && game_state!=ENDOFGAME && game_state!=PROCESSING)
    //{    
	printf("number of chances in detect:%d\n",numchances);
        numberOfFrames = GOALS;
    	while(numberOfFrames > 0)
    	{
		
	    pthread_mutex_lock(&mutex_locker);
	    pthread_cond_wait(&cond_locker,&mutex_locker);
	    
	    clock_gettime(CLOCK_REALTIME, &start_time);
	   
	    
	   
	    Rect roi2(40,30,240,180);
	    
	    crop_ROI2 = gray(roi2);
	    cvtColor(crop_ROI2, copied2, CV_BGR2GRAY);

	   
	    Rect roi1(80,60,160,120);
	    crop_ROI1 = gray(roi1);

	    cvtColor(crop_ROI1, copied, CV_BGR2GRAY);       

	
	    HoughCircles(copied2, circles2, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 36, 0, 0); //- computer 1
	    
	    if(circles2.size()!=0) 
	    {
		detect2 = 1;
	    }
	
	    HoughCircles(copied, circles1, CV_HOUGH_GRADIENT, 1, gray.rows/8, 100, 36, 0, 0); //- computer 1
	    
	    if(circles1.size()!=0) 
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
       	    
	    
	    if(!detect1 && !detect2)//Invalid
	    {
		lookup[2]++;
	    }
	    else if(detect1 && detect2) //Goal
	    {
		lookup[0]++;
	  	detect1 = detect2 = 0 ;
	    }
	    else if(detect2 && !detect1) //No goal
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
	    //printf("\n Reached here in ball detection with Frames %d \n",numberOfFrames);
	    
	    avgWCET += (final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000;
	    if(maxWCET < ((final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000))
	    	maxWCET = (final_time.tv_nsec)/NSEC_PER_MSEC + (final_time.tv_sec)/1000;
	
	    start_time.tv_sec = start_time.tv_nsec  = 0;
    	    stop_time.tv_sec = stop_time.tv_nsec = 0; 
    	    final_time.tv_sec = final_time.tv_nsec = 0;
	}//end of whilefor numframes
	
        //pthread_mutex_lock(&mutex_locker_results);
	game_state = PROCESSING;
	if((lookup[0])>80) 
        {
	    goal_status = GOAL;
        }
    	else if((lookup[1])>80) 
    	{
	    goal_status = NOGOAL;
        }
    	else if((lookup[2]-100)>80) 
	{
 	    goal_status = INVALID;
	}
        //pthread_mutex_unlock(&mutex_locker_results);
	sem_post(&sem3);
        //pthread_cond_signal(&cond_locker_results);
	printf("\n[Ball Detection]ACET over %d frames:%ldmsec\n",GOALS,avgWCET/GOALS);
	printf("\n[Ball Detection]WCET over %d frames:%ldmsec\n",GOALS,maxWCET);
	
	printf("Goals:%d,No Goals:%d,Invalid:%d,Total:%d\n",lookup[0],lookup[1],(lookup[2]),(lookup[0]+lookup[1]+lookup[2]));
	lookup[0] = lookup[1] = lookup[2] =0;
	cvDestroyWindow("ROI2_Image");
	cvDestroyWindow("ROI1_Image");
      //}//end of game state while
     }//end of numchances
	pthread_exit(NULL);
}

void* BallCapture(void *arg)
{
   /* cpu_set_t thread1_cpuset;
    CPU_ZERO(&thread1_cpuset);
    CPU_SET(0, &thread1_cpuset);
    sched_setaffinity(0,sizeof(thread1_cpuset),&thread1_cpuset);*/
    static struct timespec start_time = {0, 0};
    static struct timespec stop_time = {0, 0};
    static struct timespec final_time = {0, 0};
    unsigned long avgWCET, maxWCET = 0;
    /*cvNamedWindow(timg_window_name_houghline_eliptical, CV_WINDOW_AUTOSIZE);
    
    capture = (CvCapture *)cvCreateCameraCapture(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);*/
	
    //Start Timer
   // clock_gettime(CLOCK_REALTIME, &start_time);
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
   while(numchances>=0)
   {
   printf("number of chances in capture:%d\n",numchances);
   /*printf("Game State in Capture Thread %d\n", game_state);
   while(game_state == PROCESSING || game_state == WAIT)
   {
	usleep(50);   
   }*/

   printf("Game State in Detection Thread %d\n", game_state);
   sem_wait(&sem1);
  if(game_state == ENDOFGAME)
	break;
   cvNamedWindow(timg_window_name_houghline_eliptical, CV_WINDOW_AUTOSIZE);
    
    capture = (CvCapture *)cvCreateCameraCapture(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
   //while((game_state == START) || (game_state == CONTINUE))//while(game_state != WAIT && game_state != ENDOFGAME && game_state != PROCESSING)
   //while(game_state != WAIT && game_state != ENDOFGAME && game_state != PROCESSING)
    //{
        numberOfFrames = GOALS;
   	while(numberOfFrames > 0)
   	{
        	pthread_mutex_lock(&mutex_locker);
		clock_gettime(CLOCK_REALTIME, &start_time);

		frame=cvQueryFrame(capture);

		Mat mat_frame(frame);
		Point center_of_frame(160,120);
		circle(mat_frame, center_of_frame, 1, Scalar(0,255,0), -1, 8, 0 );
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
   	}//WHile numframes
	printf("\n[Ball Capture]ACET over %d frames:%ldmsec\n",GOALS,avgWCET/GOALS);
    	printf("\n[Ball Capture]WCET over %d frames:%ldmsec\n",GOALS,maxWCET);
        cvReleaseCapture(&capture);
        cvDestroyWindow(timg_window_name_houghline_eliptical);
   //}//while gamestate
   }//while numchances
    
   // cvReleaseCapture(&capture);
   // cvDestroyWindow(timg_window_name_houghline_eliptical);
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
    output = AUDIO_OUTPUT_PLAYBACK;
    int I, Run = 1, L;
    espeak_Initialize(output, Buflength, path, Options );
    const char *langNativeString = "en-us"; //Default to US English
    espeak_VOICE voice;
    memset(&voice, 0, sizeof(espeak_VOICE)); // Zero out the voice first
    voice.languages = langNativeString;
    voice.name = "whisperf";
    voice.variant = 2;
    espeak_SetVoiceByProperties(&voice); 
    

    pthread_attr_init(&capture_attr);
    pthread_attr_init(&detect_attr); 
    pthread_attr_init(&results_attr);

    int i, rc, scope;
    cpu_set_t threadcpu;
    int rt_max_prio, rt_min_prio;
    struct sched_param capture_prio,detect_prio,results_prio;
    struct sched_param main_param;
    pthread_attr_t main_attr;
    pid_t mainpid;
    cpu_set_t allcpuset;
    struct timeval current_time_val,start_time_val;
    printf("Starting Referee Demo\n");
    gettimeofday(&start_time_val, (struct timezone *)0);
    gettimeofday(&current_time_val, (struct timezone *)0);
   

   
  

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

   
    rc=pthread_attr_setinheritsched(&capture_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&capture_attr, SCHED_FIFO);

    rc=pthread_attr_setinheritsched(&detect_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&detect_attr, SCHED_FIFO);

    rc=pthread_attr_setinheritsched(&results_attr, PTHREAD_EXPLICIT_SCHED);
    rc=pthread_attr_setschedpolicy(&results_attr, SCHED_FIFO);

    capture_prio.sched_priority=rt_max_prio-1;
    pthread_attr_setschedparam(&capture_attr, &capture_prio);

    detect_prio.sched_priority=rt_max_prio-2;
    pthread_attr_setschedparam(&detect_attr, &detect_prio);
	

    results_prio.sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&results_attr, &results_prio);
    
    
    //printf("Threads will run on %d CPU cores\n", CPU_COUNT(&threadcpu));
    printf("Thread Priorities\n[Ball Capture]		%d\n[Ball Detection]	%d\n[Speaker]		%d\n",capture_prio.sched_priority,detect_prio.sched_priority,results_prio.sched_priority);
   
    size_speak = strlen("Start")+1;
    //printf("Saying '%s'","Start");
    espeak_Synth("Start", size_speak, position, position_type, end_position, flags,
    unique_identifier, user_data );
    
    espeak_Synchronize( );
    sem_init(&sem1,0,0);
    sem_init(&sem2,0,0);
    sem_init(&sem3,0,0);
    sem_post(&sem1);
    sem_post(&sem2);
    if (pthread_create(&captureThread, &capture_attr,BallCapture, NULL) != 0)
    {
        printf("Failed to create ball detect thread\n");
    } 
    if (pthread_create(&detectThread, &detect_attr,BallDetection, NULL) != 0)
    {
        printf("Failed to create ball detect thread\n");
    }
    if (pthread_create(&resultsThread, &results_attr,ProcessResults, NULL) != 0)
    {
        printf("Failed to create ball detect thread\n");
    }

    pthread_join(captureThread,NULL);
    pthread_join(detectThread,NULL);
    pthread_join(resultsThread,NULL);

    //printf("Goals:%d,No Goals:%d,Invalid:%d,Total:%d\n",lookup[0],lookup[1],(lookup[2]),(lookup[0]+lookup[1]+lookup[2]));
    /*if((lookup[0])>80) 
    {
	size_speak = strlen(text_goal)+1;
        printf("Saying '%s'",text_goal);
        espeak_Synth( text_goal, size_speak, position, position_type, end_position, flags,
        unique_identifier, user_data );
        espeak_Synchronize( );
        printf("\n:Done\n");  
    }
    else if((lookup[1])>80) 
    {
	size_speak = strlen(text_nogoal)+1;
        printf("Saying '%s'",text_nogoal);
        espeak_Synth(text_nogoal, size_speak, position, position_type, end_position, flags,
        unique_identifier, user_data );
        espeak_Synchronize( );
        printf("\n:Done\n");  
    }
    else if((lookup[2]-100)>80) 
    {
	size_speak = strlen(text_invalid)+1;
        printf("Saying '%s'",text_invalid);
        espeak_Synth( text_invalid, size_speak, position, position_type, end_position, flags,
        unique_identifier, user_data );
        espeak_Synchronize( );
        printf("\n:Done\n");  
    }*/
}

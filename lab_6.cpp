#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/video.hpp>

#include <vector>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace cv ;

int main( int , char** )
{
	VideoCapture cap( "001-bg-01-090.avi" ) ;								//Video capturing from video file.

	//Check the capturing is succeeded or not.
	if( !cap.isOpened( ) )
	{
		return -1 ;
	}

	namedWindow( "Show" , 1 ) ;									//Create a window for display the video.

	Mat capture , foreground ,result ;							//Declaration of capture and oreground capture.
	BackgroundSubtractorMOG mog ;

	int nowFrame = 1 ;													//Declaration a integer variable as 1st frame.
	int totalFrame = ( int )cap.get( CV_CAP_PROP_FRAME_COUNT ) ;		//Get the total frame number of the video.

	char buffer[ 10 ] ;											//Declaration of a nameing buffer.
	string fileName ;											//Declaration of file name variable in string type.

	vector< vector< Point > > contours ;
	int posX , posY , top , left , button , right , faceMin ;
	int width , height , count = 0 ;
	Mat imgROI , origForeground ;
	Mat preImg( 210 , 120 , 0 , Scalar( 0 ) ) , diffImg , meiImg( 210 , 120 , 0 , Scalar( 0 ) ) ;

	//Video processing section.
	while( nowFrame ++ != totalFrame )
	{
		Mat alignedImg( 210 , 120 , 0 , Scalar( 0 ) ) , alignedROI ;

		//Terminal the program when key pressed.
		if( waitKey( 30 ) >= 0 )
		{
			break ;
		}

		cap >> capture ;										//Get a new frame from video capture.
		imshow( "Show" , capture ) ;							//Show the captured image from video.

		//Extract the foreground of the video, and convert it into a binary image.
		mog( capture , foreground , 0.01 ) ;
		threshold( foreground , foreground , 100 , 255 , THRESH_BINARY ) ;

		dilate( foreground , foreground , Mat( ) , Point( -1 , -1 ) , 2 ) ;				//Using the filter to dilate the foreground.
		foreground.copyTo( origForeground ) ;

		//Find the contours of the foreground.
		findContours( foreground , contours , CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE ) ;

		//For each point in contours vector.
		for( int i = 0 ; i < contours.size( ) ; i ++ )
		{
			posX = 0 , posY = 0 ;
			top = INT_MAX , left = INT_MAX , button = 0 , right = 0 ;					//Set the default value of the contour image.

			//Find the area of foreground ROI.
			for( int j = 0 ; j < contours[ i ].size( ) ; j ++ )
			{
				posX = contours[ i ][ j ].x ;
				posY = contours[ i ][ j ].y ;

				if( posY < top )
				{
					top = posY ;
				}
				if( posY > button )
				{
					button = posY ;
				}
				if( posX < left )
				{
					left = posX ;
				}
				if( posX > right )
				{
					right = posX ;
				}
			}

			//Width and height of foreground ROI.
			width = right - left ;
			height = button - top ;

			//Set the image ROI.
			imgROI = origForeground( Rect( left , top , width , height ) ) ;

			//The frame that human between two base line.
			if( nowFrame >= 53 && nowFrame <= 75 )
			{
				faceMin = INT_MAX ;

				//Find the center of human head.
				for( int k = 0 ; k < contours[ i ].size( ) ; k ++ )
				{
					posY = contours[ i ][ k ].y ;

					if( posY <= top )
					{
						if( posX < faceMin )
						{
							faceMin = posX ;
						}
					}
				}

				//Position and scaling the different size foreground images.
				alignedROI = alignedImg( Rect( ( ( alignedImg.cols / 2 ) - ( faceMin + 10 - left ) ) , 30 , imgROI.cols , imgROI.rows ) ) ;
				imgROI.copyTo( alignedROI ) ;
				imshow( "Aligned image" , alignedImg ) ;

				//Save the normallized foreground image.
				fileName = itoa( nowFrame , buffer , 10 ) ;
				fileName += "_aligned.jpg" ;
				imwrite( fileName , alignedImg ) ;

				//Subtract the difference from two foreground image.
				absdiff( preImg , alignedImg , diffImg ) ;
				imshow( "Difference image" , diffImg ) ;

				//Set the previous image of next step.
				alignedImg.copyTo( preImg ) ;

				//Add an image to another.
				if( nowFrame > 53 )
				{
					add( diffImg , meiImg , meiImg ) ;
					imshow( "meiImg" , meiImg ) ;

					//Save the difference image.
					fileName = itoa( nowFrame , buffer , 10 ) ;
					fileName += "_diff.jpg" ;
					imwrite( fileName , diffImg ) ;
				}
			}
			imshow( "alignedImg" , alignedImg ) ;
		}
	}
	//Save the MEI image.
	imwrite( "meiImg.jpg" , meiImg ) ;

	cap.release( ) ;												//Release the video capture.
	return 0 ;
}

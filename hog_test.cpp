#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp> 
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/ml.hpp"
#include <ctype.h>
#include <math.h>
//#include <windows.h>
#include <sys/stat.h>
#include <sys/types.h>
//#include<io.h>
#include <fstream>

using namespace std;
using namespace cv;


Mat get_hogdescriptor_visual_image(Mat& origImg,
	vector<float>& descriptorValues, //hog feature vector
	 Size winSize, // ​​picture window size
	Size cellSize,
	 int scaleFactor, // ​​scale the proportion of the background image
	 double viz_factor)//scaling the line length ratio of the hog feature
{
	 Mat visual_image; // finally visualized image size
	resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
 
	int gradientBinSize = 9;
	// dividing 180° into 9 bins, how large (in rad) is one bin?
	 float radRangeForOneBin = 3.14 / (float)gradientBinSize; //pi=3.14 corresponds to 180°
 
	// prepare data structure: 9 orientation / gradient strenghts for each cell
	 int cells_in_x_dir = origImg.rows / cellSize.width; // number of cells in the x direction
	 int cells_in_y_dir = origImg.cols / cellSize.height;//number of cells in the y direction
	 int totalnrofcells = cells_in_x_dir * cells_in_y_dir; // total number of cells
	 // Note the definition of the three-dimensional array here
	//int ***b;
	//int a[2][3][4];
	//int (*b)[3][4] = a;
	//gradientStrengths[cells_in_y_dir][cells_in_x_dir][9]
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;
 
			for (int bin = 0; bin<gradientBinSize; bin++)
				 gradientStrengths[y][x][bin] = 0.0;//Initialize the gradient strength corresponding to the 9 bins of each cell to 0
		}
	}
 
	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	 //equivalent to blockstride = (8,8)
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;
 
	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;
 
	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}
 
				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;
 
					 gradientStrengths[celly][cellx][bin] += gradientStrength;//because C is stored in rows
 
				} // for (all bins)
 
 
				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				 cellUpdateCounter[celly][cellx]++;//Because there is overlap between blocks, it is necessary to record which cells are calculated multiple times.
 
			} // for (all cells)
 
 
		} // for (all block x pos)
	} // for (all block y pos)
 
 
	// compute average gradient strengths
	for (int celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
 
			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}
 
 
	cout << "winSize = " << winSize << endl;
	cout << "cellSize = " << cellSize << endl;
	cout << "blockSize = " << cellSize * 2 << endl;
	cout << "blockNum = " << blocks_in_x_dir << "×" << blocks_in_y_dir << endl;
	cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
 
	// draw cells
	for (int celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;
 
			int mx = drawX + cellSize.width / 2;
			int my = drawY + cellSize.height /2;
 
			rectangle(visual_image,
				Point(drawX*scaleFactor, drawY*scaleFactor),
				Point((drawX + cellSize.width)*scaleFactor,
				(drawY + cellSize.height)*scaleFactor),
				 CV_RGB (0, 0, 0), // ​​cell frame color
				1);
 
			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
				// no line to draw?
				if (currentGradStrength == 0)
					continue;
 
				 float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2; // take the intermediate value in each bin, such as 10 °, 30 °, ..., 170 °.
 
				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = cellSize.width / 2;
				float scale = viz_factor; // just a visual_imagealization scale,
				// to see the lines better
 
				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
				// draw gradient visual_imagealization
				line(visual_image,
					Point(x1*scaleFactor, y1*scaleFactor),
					Point(x2*scaleFactor, y2*scaleFactor),
					 CV_RGB (255, 255, 255), // ​​HOG visualized cell color
					1);
 
			} // for (all bins)
 
		} // for (cellx)
	} // for (celly)
 
 
	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;
 
	 return visual_image;//return the final HOG visualization image
 
}

using namespace std;
using namespace cv;

int main( int argc, char** argv ) {
  
  cv::Mat img;
  img = cv::imread("zena.png", 1);
  
  if(! img.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
    
  
  
  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
  //cv::imshow( "Display window", img );
  
  resize(img, img, Size(256, 256));
  cv::imshow( "Display window", img );
  HOGDescriptor hog;
  std::vector<float> descriptors;

	hog.winSize = Size(256, 256);
	hog.blockSize = Size(16, 16);
	hog.cellSize = Size(8, 8);
	hog.compute(img, descriptors, Size(8, 8));
	Mat background = Mat::zeros(Size(256, 256),CV_8UC1);
	Mat d = get_hogdescriptor_visual_image(background,descriptors,hog.winSize,hog.cellSize,3, 2.5);
	cout<<descriptors.size()<<endl;
	cout<<img.cols/d.cols<<endl;
	cout<<img.rows/d.rows<<endl;
	imshow("Hog of Lena",d);
	imwrite("hogvisualize.jpg",d);
  
  cv::waitKey(0);
  return 0;
}

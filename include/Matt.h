#ifndef MATT_H
#define MATT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <math.h>

#include "Neural.h"

using namespace std;
using namespace cv;

class Matt{
    public:
        // Methods
        Matt(int rows, int cols, int widthDescriptor = 90, int heightDescriptor = 45);
        Matt(Mat &img, int widthDescriptor = 90, int heightDescriptor = 45);
        virtual ~Matt();
        void intencityImage(Mat img);
        Mat getImage();
        vector<double> calculateLBPU();
        Mat histogram();
        void GenerateLbpCode();
        static Matt* CreateDetector(Mat img, int stepX = 1, int stepY = 1, int widthDescriptor = 90, int heightDescriptor = 45);
        vector< vector<double> > FindTemplates(Neural *neural);
        vector< pair<Point,Point > > MarkTemplates(Neural *neural);
        void MarkTemplates(Neural *neural,Mat &img);

        // Attributes
        int rows;
        int cols;
    protected:
    private:
        Matt(int stepX = 1, int stepY = 1, int widthDescriptor = 90, int heightDescriptor = 45, bool isDetector = true);
        void GenerateResolutionImagesLBP(Mat &img);
        int toPos(int row, int col);
        pair<int,int> toPixelPos(int pos);

        // Attributes
        vector< vector<int> > matt; // LBP images
        vector< pair<int,int> > mattSizes; // size of every level image
        vector<int> mat; // main image
        int valueU[256];
        int dir[8][2];
        int uniformU;
        int stepX;
        int stepY;
        int widthDescriptor;
        int heightDescriptor;
        bool isDetector;
};

#endif // MATT_H

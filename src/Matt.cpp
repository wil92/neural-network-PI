#include "../include/Matt.h"

Matt::Matt(int stepX, int stepY, int widthDescriptor, int heightDescriptor, bool isDetector)
{
    generateLBPCode();

    this->stepX = stepX;
    this->stepY = stepY;

    this->widthDescriptor = widthDescriptor;
    this->heightDescriptor = heightDescriptor;

    this->isDetector = isDetector;
}

Matt::Matt(int rows, int cols, int widthDescriptor, int heightDescriptor)
{
    this->cols = cols;
    this->rows = rows;

    this->mat = vector<int>(rows * cols);

    this->stepX = 1;
    this->stepY = 1;

    this->widthDescriptor = widthDescriptor;
    this->heightDescriptor = heightDescriptor;

    generateLBPCode();
}

Matt::Matt(Mat &img, int widthDescriptor, int heightDescriptor)
{
    cols = img.cols;
    rows = img.rows;

    mat = vector<int>(rows * cols);

    this->stepX = 1;
    this->stepY = 1;

    this->widthDescriptor = widthDescriptor;
    this->heightDescriptor = heightDescriptor;

    generateLBPCode();

    isDetector = false;

    intencityImage(img);
}

Matt::~Matt()
{
}

/**
 * Precalculation of the uniform patterns
 */
void Matt::generateLBPCode()
{
    int cv = 0, ct = 0;
    for (int i = 0; i <= 255; i++)
    {
        ct = 0;
        for (int j = 0; j < 7; j++)
        {
            bool s1 = (i & (1 << j)), s2 = (i & (1 << (j + 1)));
            if (s1 != s2)
            {
                ct++;
            }
        }
        if (ct == 0 || ct == 1 || ct == 2) // || ct==3
        {
            valueU[i] = cv++;
        }
        else
        {
            valueU[i] = -1;
        }
    }
    for (int i = 0; i < 255; i++)
    {
        if (valueU[i] == -1)
        {
            valueU[i] = cv;
        }
    }
    uniformU = cv + 1;
}

/**
 * Generate an instance with the resulting lbp of the image
 */
Matt *Matt::createDetector(Mat img, int stepX, int stepY, int widthDescriptor, int heightDescriptor)
{
    Matt *ret = new Matt(stepX, stepY, widthDescriptor, heightDescriptor, true);
    resize(img, img, Size(800, 800 * img.rows / img.cols));
    ret->generateResolutionImagesLBP(img);
    return ret;
}

/**
 * Given an image, return the resulting LBP 
 */
void Matt::generateResolutionImagesLBP(Mat &img)
{
    rows = img.rows;
    cols = img.cols;
    int h = rows, w = cols, cont = 0;
    int th = h, tw = w;

    // Calculating the intensity and LBP images of the pyramid
    while (1)
    {
        intencityImage(img);

        vector<int> tmpVector = vector<int>(h * w);
        mattSizes.push_back(pair<int, int>(w, h));

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int valueTmp = 0;
                for (int k = 0; k < 8; k++)
                {
                    int ni = i + dir[k][0];
                    int nj = j + dir[k][1];

                    ni = ni < 0 ? ni + 2 : ni == rows ? ni - 2 : ni;
                    nj = nj < 0 ? nj + 2 : nj == cols ? nj - 2 : nj;

                    valueTmp += matt[cont][toPos(i, j)] < matt[cont][toPos(ni, nj)] ? (1 << k) : 0;
                }
                tmpVector[toPos(i, j)] = valueTmp;
            }
        }
        matt[cont] = tmpVector;

        h = h * 3 / 4;
        w = w * 3 / 4;
        if (h < heightDescriptor || w < widthDescriptor)
            break;
        cont++;
        //cout<<cont<<endl;
        resize(img, img, Size(w, h));
    }

    rows = th;
    cols = tw;
}

vector<vector<double>> Matt::findTemplates(Neural *neural)
{

    vector<vector<double>> ret;

    int irows = rows, icols = cols;

    // Extract and check the templates
    for (int i = 0; i < matt.size(); i++)
    {
        double w = mattSizes[i].first, h = mattSizes[i].second;
        rows = h, cols = w;
        for (int j = 0; j < h - heightDescriptor + 1; j += stepY)
        {
            for (int k = 0; k < w - widthDescriptor + 1; k += stepX)
            {
                vector<double> vecTmp = vector<double>(uniformU * 4);
                for (int l = 0; l < uniformU * 4; l++)
                    vecTmp[l] = 0;

                for (int l = 0; l < heightDescriptor; l++)
                {
                    for (int z = 0; z < widthDescriptor; z++)
                    {
                        int tt = matt[i][toPos(j + l, k + z)];
                        if (l < heightDescriptor / 2 && z < widthDescriptor / 2)
                            vecTmp[valueU[tt]]++;
                        else if (l < heightDescriptor / 2 && z >= widthDescriptor / 2)
                            vecTmp[valueU[tt] + uniformU]++;
                        else if (l >= heightDescriptor / 2 && z < widthDescriptor / 2)
                            vecTmp[valueU[tt] + uniformU * 2]++;
                        else
                            vecTmp[valueU[tt] + uniformU * 3]++;
                        //                        vecTmp[ valueU[ tt ] ]++;
                    }
                }
                //                cout<<"e\n";
                if (neural->clasificate(vecTmp))
                {
                    ret.push_back(vecTmp);
                }
                //                cout<<"s\n";
            }
        }
    }
    rows = irows, cols = icols;

    return ret;
}

vector<pair<Point, Point>> Matt::markTemplates(Neural *neural)
{

    vector<pair<Point, Point>> ret;

    // Extract and check the templates
    int irows = rows, icols = cols;
    for (int i = matt.size() - 1; i >= 0; i--)
    {
        double w = mattSizes[i].first, h = mattSizes[i].second;
        rows = h, cols = w;
        for (int j = 0; j < (int)h - heightDescriptor + 1; j += stepY)
        {
            for (int k = 0; k < (int)w - widthDescriptor + 1; k += stepX)
            {
                vector<double> vecTmp = vector<double>(uniformU * 4);
                for (int l = uniformU * 4 - 1; l >= 0; l--)
                    vecTmp[l] = 0;

                for (int l = 0; l < heightDescriptor; l++)
                {
                    for (int z = 0; z < widthDescriptor; z++)
                    {
                        int tt = matt[i][toPos(j + l, k + z)];

                        if (l < heightDescriptor / 2 && z < widthDescriptor / 2)
                            vecTmp[valueU[tt]]++;
                        else if (l < heightDescriptor / 2 && z >= widthDescriptor / 2)
                            vecTmp[valueU[tt] + uniformU]++;
                        else if (l >= heightDescriptor / 2 && z < widthDescriptor / 2)
                            vecTmp[valueU[tt] + uniformU * 2]++;
                        else
                            vecTmp[valueU[tt] + uniformU * 3]++;
                        //                        vecTmp[ valueU[ tt ] ]++;
                    }
                }
                if (neural->clasificate(vecTmp))
                {
                    double x = k;
                    double y = j;
                    ret.push_back(make_pair(Point((x)*icols / w, irows - (y)*irows / h),
                                            Point((x + widthDescriptor) * icols / w, irows - (y + heightDescriptor) * irows / h)));
                }
            }
        }
    }
    rows = irows, cols = icols;

    return ret;
}

void Matt::markTemplates(Neural *neural, Mat &img)
{
    vector<pair<Point, Point>> lis = markTemplates(neural);
    for (int i = 0; i < lis.size(); i++)
    {
        rectangle(img,
                  Point((lis[i].first.x) * img.cols / cols, img.rows - (lis[i].first.y) * img.rows / rows),
                  Point((lis[i].second.x) * img.cols / cols, img.rows - (lis[i].second.y) * img.rows / rows),
                  Scalar(255, 0, 0));
    }
}

void Matt::intencityImage(Mat img)
{

    rows = img.rows;
    cols = img.cols;

    mat = vector<int>(rows * cols);

    int maxValue = 0, it2 = 0;

    if (img.channels() == 1)
    {
        Mat_<uchar>::iterator it = img.begin<uchar>();
        Mat_<uchar>::iterator eit = img.end<uchar>();
        for (it2 = 0, it = img.begin<uchar>(); it != eit; it++, it2++)
        {
            mat[it2] = (int)((*it));
        }
        if (isDetector)
        {
            matt.push_back(mat);
        }
        return;
    }

    Mat_<Vec3b>::iterator it = img.begin<Vec3b>();
    Mat_<Vec3b>::iterator eit = img.end<Vec3b>();

    for (; it != eit; it++)
    {
        maxValue = max((*it)[0] + (*it)[1] + (*it)[2], maxValue);
    }

    if (!maxValue)
        maxValue = 1;

    for (it2 = 0, it = img.begin<Vec3b>(); it != eit; it++, it2++)
    {
        mat[it2] = (int)((*it)[0] + (*it)[1] + (*it)[2]) * 255 / maxValue;
    }

    if (isDetector)
    {
        matt.push_back(mat);
    }
}

Mat Matt::getImage()
{
    Mat a;
    a.create(rows, cols, 0);

    if (isDetector)
        return a;

    for (int it2 = 0, tam = rows * cols; it2 < tam; it2++)
    {
        pair<int, int> p = Matt::toPixelPos(it2);
        a.at<uchar>(p.first, p.second) = mat[it2];
    }

    return a;
}

vector<double> Matt::calculateLBPU()
{

    //Tama�o del descriptor
    vector<double> ret = vector<double>(uniformU * 4);

    for (int i = 0; i < uniformU; i++)
        ret[i] = 0;

    if (isDetector)
        return ret;

    int nc = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int value = 0;
            for (int k = 0; k < 8; k++)
            {
                int ni = i + dir[k][0];
                int nj = j + dir[k][1];

                ni = ni < 0 ? ni + 2 : ni == rows ? ni - 2 : ni;
                nj = nj < 0 ? nj + 2 : nj == cols ? nj - 2 : nj;

                value += mat[toPos(i, j)] < mat[toPos(ni, nj)] ? (1 << k) : 0;
            }

            if (i < rows / 2 && j < cols / 2)
                ret[valueU[value]]++;
            else if (i < rows / 2 && j >= cols / 2)
                ret[valueU[value] + uniformU]++;
            else if (i >= rows / 2 && j < cols / 2)
                ret[valueU[value] + uniformU * 2]++;
            else
                ret[valueU[value] + uniformU * 3]++;
        }
    }

    return ret;
}

Mat Matt::histogram()
{

    Mat hist(300, 512, CV_8UC3, Scalar(20, 20, 20));

    if (isDetector)
        return hist;

    double height = 300., width = 512.;

    vector<double> v = calculateLBPU();

    double values[256], maxim = 0;

    for (int i = 0; i < 256; i++)
        values[i] = 0;

    for (int i = 0; i < v.size(); i++)
    {
        values[i] = (double)v[i];
        maxim = max(maxim, values[i]);
    }

    maxim = 1000.;

    double paso = width / (double)(v.size());

    for (int i = 0; i < v.size(); i++)
    {
        double peso = values[i] * height / maxim;
        rectangle(hist,
                  Point(paso * (double)i, height - peso),
                  Point(paso * (double)(i + 1), height),
                  Scalar(255, 0, 0),
                  CV_FILLED);
    }

    return hist;
}

pair<int, int> Matt::toPixelPos(int pos)
{
    return pair<int, int>(pos / cols, pos % cols);
}

int Matt::toPos(int row, int col)
{
    return row * cols + col;
}

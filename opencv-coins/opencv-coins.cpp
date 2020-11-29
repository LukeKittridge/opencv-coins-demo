// opencv-coins.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void testCascadeClassifier() {
    CascadeClassifier tenPence;
    tenPence.load(R"(cascade.xml)");
    vector<Rect> coins;

    cout << "Enter image number 0-6:";
    int num;
    cin >> num;

    do {
        Mat testImage = imread(R"(test_images\)" + to_string(num) + ".jpg");
        tenPence.detectMultiScale(testImage, coins);

        for (Rect coin : coins)
        {
            rectangle(testImage, coin, Scalar(0, 255, 0));
        }

        imshow(to_string(num), testImage);
        num = waitKey(0) - 48;
        destroyAllWindows();
    } while (num > -1);
}

void testCascadeClassifierVideo() {
    CascadeClassifier tenPence;
    tenPence.load(R"(cascade.xml)");
    vector<Rect> coins;

    VideoCapture cap;
    cap.open(R"(test_video\coin_vid.mp4)");
    Mat frame;
    int i = 0;
    for (;;) {
    i++;
        cap.read(frame);

        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        if (i % 3 == 0) {
        tenPence.detectMultiScale(frame, coins);

        for (Rect coin : coins)
        {
            rectangle(frame, coin, Scalar(0, 255, 0));
        }

        imshow("Live", frame);
    }
    if (waitKey(5) >= 0)
        break;
    }
    
}

string doubleToString(double value) 
{
    stringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    string totalString = ss.str();
    return totalString;
}

void testHoughCircle() {

    cout << "Enter image number 0-6:";
    int num;
    cin >> num;

    do {
        Mat img = imread(R"(test_images\6.jpg)");

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        medianBlur(gray, gray, 3);

        vector<Vec3f> circles;
        HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 200, 100, 1, 20, 100);

        int i = 0;

        Vec3b intensity;

        Scalar ten = Scalar(61, 96, 145);
        Scalar twenty = Scalar(106, 152, 187);
        Scalar pound = Scalar(121, 135, 146);

        int param = 40;
        double total = 0;

        for (Vec3f c : circles)
        {
            int sumBlue = 0;
            int sumGreen = 0;
            int sumRed = 0;
            Point center = Point(c[0], c[1]);

            for (int i = center.x - param; i <= center.x + param; i++) {
                for (int j = center.y - param; j < center.y + param; j++) {
                    if (i > 0 && j > 0) {
                        intensity = img.at<Vec3b>(Point(i, j));
                        sumBlue += intensity.val[0];
                        sumGreen += intensity.val[1];
                        sumRed += intensity.val[2];
                    }
                }
            }

            int numPixels = (param * 2) * (param * 2);
            int avBlue = sumBlue / numPixels;
            int avGreen = sumGreen / numPixels;
            int avRed = sumRed / numPixels;

            double value = 0;

            int tol = 40;
            if ((avBlue <= ten[0] + tol && avBlue >= ten[0] - tol) && (avGreen <= ten[1] + tol && avGreen >= ten[1] - tol) && (avRed <= ten[2] + tol && avRed >= ten[2] - tol)) {
                value = 0.1;
            }
            else if ((avBlue <= twenty[0] + tol && avBlue >= twenty[0] - tol) && (avGreen <= twenty[1] + tol && avGreen >= twenty[1] - tol) && (avRed <= twenty[2] + tol && avRed >= twenty[2] - tol)) {
                value = 0.2;
            }
            else if ((avBlue <= pound[0] + tol && avBlue >= pound[0] - tol) && (avGreen <= pound[1] + tol && avGreen >= pound[1] - tol) && (avRed <= pound[2] + tol && avRed >= pound[2] - tol))
            {
                value = 1;
            }

            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];

            cout << i << ":Blue: " << to_string(blue) << " Green: " << to_string(green) << " Red: " << to_string(red) << "\n";
            cout << i << ":avBlue: " << to_string(avBlue) << " avGreen: " << to_string(avGreen) << " avRed: " << to_string(avRed) << "\n";

            putText(img, to_string(i) + ": " + doubleToString(value), center, FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 2);
            circle(img, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
            int radius = c[2];
            circle(img, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
            i++;
            total += value;
        }

        putText(img, "Total: " + doubleToString(total), Point(0, img.size().height), FONT_HERSHEY_PLAIN, 5, Scalar(0, 0, 255));
        imshow(to_string(num), img);
        waitKey();
        num = waitKey(0) - 48;
        destroyAllWindows();
    } while (num > -1);

    
}

int main()
{
    testCascadeClassifierVideo();
    //testCascadeClassifier();
    //testHoughCircle();
    
    return 0;
}

void resizeImages(string path) {
    Mat img2;
    for (const auto& entry : fs::directory_iterator(path))
    {
        Mat img1 = imread(entry.path().u8string(), IMREAD_COLOR);
        resize(img1, img2, Size(), 0.25,0.25);
        imwrite(entry.path().filename().u8string(), img2);
    }
}
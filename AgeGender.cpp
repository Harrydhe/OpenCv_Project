#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main() {
    VideoCapture video(0);
    CascadeClassifier facedetect;
    Mat img;

    // Muat model deteksi wajah
    facedetect.load("haarcascade_frontalface_default.xml");

    // Muat model deteksi umur dan gender
    string ageModel = "age_net.caffemodel";
    string ageProto = "age_deploy.prototxt";
    string genderModel = "gender_net.caffemodel";
    string genderProto = "gender_deploy.prototxt";

    Net ageNet = readNet(ageModel, ageProto);
    Net genderNet = readNet(genderModel, genderProto);

    // Label umur dan gender
    vector<string> ageList = { "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-" };
    vector<string> genderList = { "Pria", "Wanita" };

    while (true) {
        video.read(img);
        vector<Rect> faces;

        facedetect.detectMultiScale(img, faces, 1.3, 5);

        cout << "Jumlah wajah terdeteksi: " << faces.size() << endl;

        for (int i = 0; i < faces.size(); i++) {
            Rect face = faces[i];
            rectangle(img, face.tl(), face.br(), Scalar(50, 50, 255), 3);

            // Siapkan ROI wajah untuk deteksi umur dan gender
            Mat faceROI = img(face);

            // Siapkan blob dari ROI wajah
            Mat blob = blobFromImage(faceROI, 1.0, Size(227, 227), Scalar(78.4263377603, 87.7689143744, 114.895847746), false);

            // Prediksi gender
            genderNet.setInput(blob);
            Mat genderPreds = genderNet.forward();
            int genderIdx = genderPreds.at<float>(0) > genderPreds.at<float>(1) ? 0 : 1;
            string gender = genderList[genderIdx];

            // Prediksi umur
            ageNet.setInput(blob);
            Mat agePreds = ageNet.forward();
            int ageIdx = max_element(agePreds.begin<float>(), agePreds.end<float>()) - agePreds.begin<float>();
            string age = ageList[ageIdx];

            // Tampilkan umur dan gender pada gambar
            string label = gender + ", " + age;
            putText(img, label, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 2);
        }

        imshow("Frame", img);
        if (waitKey(1) == 27) {
            break; // Keluar jika menekan tombol ESC
        }
    }

    return 0;
}

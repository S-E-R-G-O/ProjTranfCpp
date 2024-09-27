#include "Processing.h"
#include <iostream>

using namespace std;

int main() {
	try {
		Processing processing("C:/Users/Main/Desktop/video1.avi", "C:/Users/Main/Desktop/Opencvtest/video2.avi");
		processing.detectedChanges();


	}
	catch (const exception& e) {
		cerr << e.what() << endl;
		return -1;
	}
}


#include <bits/stdc++.h>
#include "exponentialweaklearner.h"
#include "mseweaklearner.h"
#include "logisticweaklearner.h"
using namespace std;

template<typename T>
ostream& operator <<(ostream &out, const vector<T> &v) {
	for(const auto &item : v) out << item << '\n';
	return out;
}

class AdaBoost {
private:
	vector<WeakLearner> weakLearners;
	
	double getError(const vector<vector<double>> &data, const vector<double> &w, const WeakLearner &weakLearner) const {
		int i;
		double error;
		
		for(error = 0, i = 0; i < data.size(); ++i) {
			if(weakLearner.getLabel(data[i]) != data[i].back()) error += w[i];
		}
		
		return error;
	}
	
public:
	double getMargin(const vector<double> &instance, const WeakLearner &weakLearner) const {
		return weakLearner.getAlpha() * weakLearner.getLabel(instance);
	}
	
	int predict(const vector<double> &instance, int num) const {
		double s = 0;
		
		if(num < 0) num = weakLearners.size();
		for(int i = 0; i < weakLearners.size() && i < num; ++i) {
			s += getMargin(instance, weakLearners[i]);
			// cout << "wl " << weakLearnerLabel(instance, dim[i], threshold[i], label[i]) << ' ' << alpha[i] << endl;
		}
		
		return (s < 0 ? -1 : 1);
	}
	
	void train(const vector<vector<double>> &data, int numOfIterations, const WeakLearnerProvider *weakLearnerProvider,
			ofstream *timeFile = nullptr) {
		const int numOfFeatures = data[0].size() - 1;
		vector<double> w(data.size(), 1.0 / data.size());
		vector<double> f(data.size(), 0);
		vector<vector<int>> sortedData(data[0].size() - 1);
		
		auto trainStartTime = chrono::high_resolution_clock::now();
		
		for(int i = 0; i < data[0].size() - 1; ++i) {
			for(int j = 0; j < data.size(); ++j)
				sortedData[i].push_back(j);
			
			sort(sortedData[i].begin(), sortedData[i].end(), [&data, &i](int x, int y) {
				return data[x][i] < data[y][i];
			});
		}
		
		for(int iteration = 1; iteration <= numOfIterations; ++iteration) {
			cout << "Iteration " << iteration << endl;
			
			auto [weakLearner, stop] = weakLearnerProvider->getWeakLearner(data, sortedData, w, f);
			
			if(weakLearner.getDim() < 0) {
				cout << "Error greater than 0.5 at iteration " << iteration << endl;
				return;
			}
			
			// cout << weakLearner << '\n';
			weakLearners.push_back(weakLearner);
			
			if(stop) return;
			
			for(int i = 0; i < data.size(); ++i) {
				f[i] += weakLearner.getAlpha() * weakLearner.getLabel(data[i]);
			}
			
			weakLearnerProvider->updateWeights(weakLearner, data, w, f);
			
			if(timeFile && iteration % 1 == 0) { // print timestamp every iteration
				auto currentTime = chrono::high_resolution_clock::now();
				(*timeFile) << std::chrono::duration_cast<std::chrono::microseconds>(currentTime - trainStartTime).count() / 1000000.0 << '\n';
			}
			
			// cout << "w[]: " << w << '\n';
		}
		
		cout << "w[]: ";
		for(int i = 0; i < 20 && i < w.size(); ++i) cout << w[i] << ' ';
		cout << '\n';
		
		double sumAlpha = 0;
		for(const auto &weakLearner : weakLearners) sumAlpha += weakLearner.getAlpha();
		
		cout << "Sum alpha = " << sumAlpha << '\n';
	}

	double validate(const vector<vector<double>> &data, ofstream *fout = nullptr, int num = -1) const {
		int misclassifiedCount = 0;
		
		for(const auto &instance : data) {
			int predictedLabel = predict(instance, num);
			
			if(fout) {
				for(int i = 0; i < instance.size() - 1; ++i) (*fout) << instance[i] << ' ';
				(*fout) << predictedLabel << '\n';
			}
			
			if(predictedLabel != instance.back()) {
				++misclassifiedCount;
			}
		}
		
		// cout << "dbg " << num << ' ' << misclassifiedCount << endl;
		return 100.0 * misclassifiedCount / data.size();
	}
	
	vector<double> getErrorPlot(const vector<vector<double>> &data, int num = -1) const {
		vector<double> margins(data.size(), 0);
		vector<double> errors;
		
		if(num < 0) num = weakLearners.size();
		for(int i = 0; i < num && i < weakLearners.size(); ++i) {
			int misclassifiedCount = 0;
			
			for(int j = 0; j < data.size(); ++j) {
				margins[j] += getMargin(data[j], weakLearners[i]);
				
				int predictedLabel = (margins[j] < 0 ? -1 : 1);
				if(predictedLabel != data[j].back()) ++misclassifiedCount;
			}
			
			errors.push_back(100.0 * misclassifiedCount / data.size());
		}
		
		return errors;
	}
	
	vector<double> getMargins(const vector<vector<double>> &data, int num = -1) const {
		double sumAlpha = 0;
		vector<double> margins(data.size(), 0);
		
		if(num < 0) num = weakLearners.size();
		for(int i = 0; i < num && i < weakLearners.size(); ++i) {
			sumAlpha += weakLearners[i].getAlpha();
			
			for(int j = 0; j < data.size(); ++j)
				margins[j] += getMargin(data[j], weakLearners[i]);
		}
		
		for(int j = 0; j < data.size(); ++j) {
			margins[j] *= data[j].back();
			margins[j] /= sumAlpha;
		}
		
		return margins;
	}
};

WeakLearnerProvider* getWeakLearnerProvider(const string &lossFunction) {
	if(lossFunction == "mse") return new MSEWeakLearnerProvider();
	if(lossFunction == "logistic") return new LogisticWeakLearnerProvider();
	
	return new ExponentialWeakLearnerProvider();
}

vector<vector<double>> readFile(string filename) {
	ifstream fin(filename);
	vector<vector<double>> data;
	
	string line;
	while(getline(fin, line)) {
		double x;
		vector<double> v;
		
		istringstream iss(line);
		int cnt = 0;
		while(iss >> x) v.push_back(x), ++cnt;
		
		// v.back() = (v.back() <= 0.5 ? -1 : 1); // for banknote
		// v.back() = (v.back() <= 1.5 ? -1 : 1); // for satimage
		v.back() = (v.back() <= 5.5 ? -1 : 1); // for mnist
		
		data.push_back(v);
	}
	
	fin.close();
	
	return data;
}

void printMargins(const vector<double> &margins, ofstream &fout) {
	for(auto margin : margins) fout << margin << '\n';
}

int main(int argc, char **argv) {
	assert(argc > 2);
	
	string datasetName = argv[1];
	string lossFunction = argv[2];
	int numOfIterations = (argc > 3 ? atoi(argv[3]) : 10);
	
	string folderName = "out/" + datasetName;
	string prefix = to_string(numOfIterations) + "-" + lossFunction;
	ofstream summaryFile(folderName + "/" + prefix + "-summary.txt");
	ofstream timeFile(folderName + "/" + prefix + "-traintime.txt");
	ofstream trainPredictionFile(folderName + "/" + prefix + "-train.txt");
	ofstream testPredictionFile(folderName + "/" + prefix + "-test.txt");
	ofstream trainMarginsFile(folderName + "/" + prefix + "-trainmargins.txt");
	ofstream testMarginsFile(folderName + "/" + prefix + "-testmargins.txt");
	ofstream trainErrorPlot(folderName + "/" + prefix + "-trainerrorplot.txt");
	ofstream testErrorPlot(folderName + "/" + prefix + "-testerrorplot.txt");
	
	auto trainData = readFile("../datasets/" + datasetName + "/train.txt");
	auto testData = readFile("../datasets/" + datasetName + "/test.txt");
	
	// cerr << trainData << '\n';
	
	cout << "Dataset size: " << trainData.size() << endl;
	cout << "Number of iterations: " << numOfIterations << endl;
	
	int trainPositive = 0;
	for(const auto &instance : trainData)
		if(instance.back() > 0) ++trainPositive;
	
	int testPositive = 0;
	for(const auto &instance : testData)
		if(instance.back() > 0) ++testPositive;
	
	cout << "Percent of positive (train): " << 100.0 * trainPositive / trainData.size() << '%' << endl;
	cout << "Percent of positive (test): " << 100.0 * testPositive / testData.size() << '%' << endl;
	
	AdaBoost adaBoost;
	
	auto trainStartTime = chrono::high_resolution_clock::now();
	adaBoost.train(trainData, numOfIterations, getWeakLearnerProvider(lossFunction), &timeFile);
	auto trainStopTime = chrono::high_resolution_clock::now();
	
	double trainError = adaBoost.validate(trainData, &trainPredictionFile);
	cout << "Train error: " << trainError << endl;
	// for(int i = 0; i < numOfIterations; ++i) {
	// 	double error = adaBoost.validate(trainData, nullptr, i + 1);
	// 	if(error == 0) {
	// 		cout << "Iteration " << i + 1 << endl;
	// 		break;
	// 	}
	// 	// cout << "Train error: " << adaBoost.validate(trainData, &testPredictionFile, i + 1) << endl;
	// }
	
	double testError = adaBoost.validate(testData, &testPredictionFile);
	cout << "Test error: " << testError << endl;
	
	trainErrorPlot << adaBoost.getErrorPlot(trainData);
	testErrorPlot << adaBoost.getErrorPlot(testData);
	
	printMargins(adaBoost.getMargins(trainData), trainMarginsFile);
	printMargins(adaBoost.getMargins(testData), testMarginsFile);
	
	// create summary file
	summaryFile << "Dataset name: " << datasetName << '\n';
	summaryFile << "Number of iterations: " << numOfIterations << '\n';
	summaryFile << "Loss function: " << lossFunction << '\n';
	
	summaryFile << "Train set size: " << trainData.size() << '\n';
	summaryFile << "Percent of positive (train set): " << 100.0 * trainPositive / trainData.size() << '%' << '\n';
	summaryFile << "Train error: " << trainError << '\n';
	
	summaryFile << "Test set size: " << testData.size() << '\n';
	summaryFile << "Percent of positive (test set): " << 100.0 * testPositive / testData.size() << '%' << '\n';
	summaryFile << "Test error: " << testError << '\n';
	
	summaryFile << "Training duration: " <<
		std::chrono::duration_cast<std::chrono::microseconds>(trainStopTime - trainStartTime).count() / 1000000.0 << '\n';
	
	trainPredictionFile.close();
	testPredictionFile.close();
	trainMarginsFile.close();
	testMarginsFile.close();
	summaryFile.close();
	trainErrorPlot.close();
	testErrorPlot.close();
	
	return 0;
}

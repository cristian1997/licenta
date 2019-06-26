#include <bits/stdc++.h>
using namespace std;

class AdaBoost {
	private:
		vector<double> dim, threshold, label;
		vector<double> alpha;
		vector<vector<double>> values;
		
		int weakLearnerLabel(const vector<double> &instance, int dim, double threshold, int label) {
			return (instance[dim] <= threshold ? label : -label);
		}
		
		double getError(const vector<vector<double>> &data, vector<double> &w, int dim, double threshold, int label) {
			int i;
			double error;
			
			for(error = 0, i = 0; i < data.size(); ++i) {
				if(data[i][dim] <= threshold && data[i].back() != label) error += w[i];
				else if(data[i][dim] > threshold && data[i].back() == label) error += w[i];
			}
			
			return error;
		}
		
		int predict(const vector<double> &instance) {
			double s = 0;
			
			for(int i = 0; i < alpha.size(); ++i) {
				s += alpha[i] * weakLearnerLabel(instance, dim[i], threshold[i], label[i]);
			}
			
			return (s < 0 ? -1 : 1);
		}
		
	public:
		void train(const vector<vector<double>> &data, int numOfIterations) {
			vector<double> w;
			
			values.resize(data[0].size() - 1);
			for(const auto instance : data) {
				for(int i = 0; i < instance.size() - 1; ++i) values[i].push_back(instance[i]);
			}
			
			for(int i = 0; i < values.size(); ++i) {
				sort(values[i].begin(), values[i].end());
				values[i].erase(unique(values[i].begin(), values[i].end()), values[i].end());
			}
			
			w.assign(data.size(), 1.0 / data.size());
			
			for(int iteration = 1; iteration <= numOfIterations; ++iteration) {
				int iDim = 0, iThreshhold = 0, iLabel = 0;
				double minError = 1;
				
				for(int i = 0; i < values.size(); ++i) {
					for(auto x : values[i]) {
						double error = getError(data, w, i, x, 1);
						if(error < minError) minError = error, iDim = i, iThreshhold = x, iLabel = 1;
						
						error = getError(data, w, i, x, -1);
						// cerr << "Error: " << error << endl;
						if(error < minError) minError = error, iDim = i, iThreshhold = x, iLabel = -1;
					}
				}
				
				// cerr << "minError " << minError << endl;
				if(minError >= 0.5 || minError <= 1e-9) { cout << iteration << endl; break; }
				
				double iAlpha = log((1 - minError) / minError) / 2;
				
				alpha.push_back(iAlpha);
				dim.push_back(iDim);
				threshold.push_back(iThreshhold);
				label.push_back(iLabel);
				
				double s = 0;
				for(int i = 0; i < data.size(); ++i) {
					w[i] *= exp(-iAlpha * weakLearnerLabel(data[i], iDim, iThreshhold, iLabel) * data[i].back());
					s += w[i];
				}
				
				for(double &weight : w) weight /= s;
			}
			
			cout << "OK" << endl;
			for(int i = 0; i < alpha.size(); ++i) {
				cout << alpha[i] << ' ' << dim[i] << ' ' << threshold[i] << ' ' << label[i] << endl;
			}
			
			// cout << "w[] = ";
			// for(int i = 0; i < data.size(); ++i) cout << w[i] << ' ';
			// cout << '\n';
		}
	
	double validate(const vector<vector<double>> &data) {
		int misclassifiedCount = 0;
		
		for(const auto &instance : data) {
			if(predict(instance) != instance.back()) ++misclassifiedCount;
		}
		
		return 1.0 * misclassifiedCount / data.size();
	}
};

vector<vector<double>> readFile(string filename) {
	ifstream fin(filename);
	vector<vector<double>> data;
	
	string line;
	while(getline(fin, line)) {
		double x;
		vector<double> v;
		
		istringstream iss(line);
		while(iss >> x) v.push_back(x);
		
		data.push_back(v);
	}
	
	fin.close();
	return data;
}

int main(int argc, char **argv) {
	int numOfIterations = (argc > 1 ? atoi(argv[1]) : 10);
	
	auto trainData = readFile("../datasets/corner/train.txt");
	auto validationData = readFile("../datasets/corner/validation.txt");
	
	AdaBoost adaBoost;
	
	adaBoost.train(trainData, numOfIterations);
	cout << "Train error: " << adaBoost.validate(trainData) << endl;
	// cout << "Validation error: " << adaBoost.validate(validationData) << endl;
	
	return 0;
}
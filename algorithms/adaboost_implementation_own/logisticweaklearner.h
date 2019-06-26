#ifndef LOGISTICWEAKLEARNER_H
#define LOGISTICWEAKLEARNER_H

#include <iostream>
#include "weaklearnerprovider.h"

class LogisticWeakLearnerProvider : public WeakLearnerProvider {
public:
	std::pair<WeakLearner, bool> getWeakLearner(
			const std::vector<std::vector<double>> &data, const std::vector<std::vector<int>> &sortedData,
			const std::vector<double> &w, const std::vector<double> &f) const override {
		// std::cout << "LOGISTIC\n";
		int numOfFeatures = sortedData.size();
		int iDim = 0;
		double minError = 1, iThreshold = 0, iLabel = 0;
		
		// cout << "w[]: " << w << endl;
		
		for(int i = 0; i < numOfFeatures; ++i) {
			double leftError = 0, rightError = 0;
			// assuming instances to the left of threshold are labeled negative and rest positive
			for(int j = 0; j < data.size(); ++j) if(data[j].back() == -1) rightError += w[j];
			
			double error = leftError + rightError;
			if(error < minError) minError = error, iDim = i, iThreshold = data[sortedData[i][0]][i] - 1, iLabel = -1;
			if(1 - error < minError) minError = error, iDim = i, iThreshold = data[sortedData[i][0]][i] - 1, iLabel = 1;
			
			// cout << i << ' ' << data[sortedData[i][0]][i] - 1 << ' ' << min(error, 1 - error) << endl;
			
			for(int j = 0; j < data.size(); ++j) {
				int idx = sortedData[i][j];
				if(data[idx].back() == -1) rightError -= w[idx];
				else leftError += w[idx];
				
				if(j + 1 < data.size() && data[sortedData[i][j]][i] == data[sortedData[i][j + 1]][i]) continue;
				
				double currThreshold;
				if(j + 1 < data.size()) currThreshold = (data[sortedData[i][j]][i] + data[sortedData[i][j + 1]][i]) / 2;
				else currThreshold = data[sortedData[i][j]][i] + 1;
				
				error = leftError + rightError;
				if(error < minError) minError = error, iDim = i, iThreshold = currThreshold, iLabel = -1;
				else if(1 - error < minError) minError = 1 - error, iDim = i, iThreshold = currThreshold, iLabel = 1;
				
				// cout << i << ' ' << currThreshold << ' ' << min(error, 1 - error) << endl;
			}
		}
		
		// cout << "minError = " <<  minError << endl;
		
		// cerr << "minError " << minError << endl;
		if(minError <= 0) {
			double iAlpha = 1.0;
			return { { iDim, iThreshold, iLabel, iAlpha }, true };
		}
		
		if(minError >= 0.5) return { { -1, -1, -1, -1 }, true };
		
		double iAlpha = log((1 - minError) / minError) / 2;			
		return { { iDim, iThreshold, iLabel, iAlpha }, false };
	}
	
	void updateWeights(const WeakLearner &weakLearner, const std::vector<std::vector<double>> &data,
		std::vector<double> &w, const std::vector<double> &f) const override {
		double s = 0;
		for(int i = 0; i < data.size(); ++i) {
			w[i] = 1.0 / (1.0 + exp(data[i].back() * f[i]));
			s += w[i];
		}
		
		for(double &weight : w) weight /= s;
	}
};

#endif

#ifndef MSEWEAKLEARNER_H
#define MSEWEAKLEARNER_H

#include <iostream>
#include "weaklearnerprovider.h"

class MSEWeakLearnerProvider : public WeakLearnerProvider {
public:
	std::pair<WeakLearner, bool> getWeakLearner(
			const std::vector<std::vector<double>> &data, const std::vector<std::vector<int>> &sortedData,
			const std::vector<double> &w, const std::vector<double> &f) const override {
		// std::cout << "MSE\n";
		int numOfFeatures = sortedData.size();
		int iDim = 0;
		double iAlpha = -1e18, iThreshold = 0, iLabel = 0;
		
		// cout << "w[]: " << w << endl;
		
		for(int i = 0; i < numOfFeatures; ++i) {
			double alpha = 0;
			// assuming instances to the left of threshold are labeled negative and rest positive
			for(int j = 0; j < data.size(); ++j) alpha += data[j].back() - f[j];
			
			if(alpha > iAlpha) iAlpha = alpha, iDim = i, iThreshold = data[sortedData[i][0]][i] - 1, iLabel = -1;
			if(-alpha > iAlpha) iAlpha = -alpha, iDim = i, iThreshold = data[sortedData[i][0]][i] - 1, iLabel = 1;
			
			// cout << i << ' ' << iThreshold << ' ' << alpha << endl;
			
			for(int j = 0; j < data.size(); ++j) {
				int idx = sortedData[i][j];
				alpha -= 2 * (data[idx].back() - f[idx]);
				
				if(j + 1 < data.size() && data[sortedData[i][j]][i] == data[sortedData[i][j + 1]][i]) continue;
				
				double currThreshold;
				if(j + 1 < data.size()) currThreshold = (data[sortedData[i][j]][i] + data[sortedData[i][j + 1]][i]) / 2;
				else currThreshold = data[sortedData[i][j]][i] + 1;
				
				// cerr << i << ' ' << currThreshold << ' ' << alpha << endl;
				if(alpha > iAlpha) iAlpha = alpha, iDim = i, iThreshold = currThreshold, iLabel = -1;
				if(-alpha > iAlpha) iAlpha = -alpha, iDim = i, iThreshold = currThreshold, iLabel = 1;
			}
		}
		
		return { { iDim, iThreshold, iLabel, iAlpha / data.size() }, false };
	}
	
	void updateWeights(const WeakLearner &weakLearner, const std::vector<std::vector<double>> &data,
		std::vector<double> &w, const std::vector<double> &f) const override {
		
	}
};

#endif

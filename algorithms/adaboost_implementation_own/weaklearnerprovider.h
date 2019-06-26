#ifndef WEAKLEARNERPROVIDER_H
#define WEAKLEARNERPROVIDER_H

#include <vector>
#include "weaklearner.h"

class WeakLearnerProvider {
public:
	virtual std::pair<WeakLearner, bool> getWeakLearner(
		const std::vector<std::vector<double>> &data,
		const std::vector<std::vector<int>> &sortedData,
		const std::vector<double> &w,
		const std::vector<double> &f) const = 0;
	
	virtual void updateWeights(
		const WeakLearner &weakLearner,
		const std::vector<std::vector<double>> &data,
		std::vector<double> &w,
		const std::vector<double> &f
		) const = 0;
};

#endif

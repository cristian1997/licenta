#ifndef WEAKLEARNER_H
#define WEAKLEARNER_H

#include <vector>

class WeakLearner {
private:
	int dim;
	double threshold, label, alpha;

public:
	WeakLearner(int dim, double threshold, double label, double alpha)
		: dim(dim), threshold(threshold), label(label), alpha(alpha) {}
	
	double getLabel(const std::vector<double> &instance) const {
		return (instance[dim] <= threshold ? label : -label);
	}
	
	int getDim() const { return dim; }
	double getThreshold() const { return threshold; }
	double getLabel() const { return label; }
	double getAlpha() const { return alpha; }
	
	void setAlpha(double newAlpha) { alpha = newAlpha; }
	
	friend std::ostream& operator <<(std::ostream &out, const WeakLearner &weakLearner) {
		out << "*************************\n";
		out << "dim " << weakLearner.dim << '\n';
		out << "threshold " << weakLearner.threshold << '\n';
		out << "label " << weakLearner.label << '\n';
		out << "alpha " << weakLearner.alpha << '\n';
		return out << "*************************";
	}
};

#endif

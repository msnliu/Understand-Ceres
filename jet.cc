#include<iostream>
#include<ceres/ceres.h>

using namespace std;
using namespace ceres;

const int kNumObservations = 14;
const double data[] = {
  0, 1.057,
  0.22, 1.101,
  0.33, 1.125,
  0.44, 1.151,
  0.55, 1.181,
  0.66, 1.213,
  0.77, 1.249,
  0.88, 1.291,
  0.99, 1.34,
  1.1, 1.399,
  1.2, 1.466,
  1.3, 1.555,
  1.4, 1.694,
  1.5, 2.5,
};

struct Rat43CostFunctor {
  Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T* parameters, T* residuals) const {
    cout << "here" << endl;
    const T b1 = parameters[0];
    const T b2 = parameters[1];
    const T b3 = parameters[2];
    const T b4 = parameters[3];
    residuals[0] = b1 * pow(1.0 + exp(b2 -  b3 * x_), -1.0 / b4) - y_;
    return true;
  }

  private:
    const double x_;
    const double y_;
};

class Rat43Automatic : public ceres::SizedCostFunction<1,4> {
 public:
  Rat43Automatic(const Rat43CostFunctor* functor) : functor_(functor) {}
  virtual ~Rat43Automatic() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    // Just evaluate the residuals if Jacobians are not required.
    // cout << 1 << endl;
    if (!jacobians) return (*functor_)(parameters[0], residuals);

    // Initialize the Jets
    ceres::Jet<double, 4> jets[4];
    for (int i = 0; i < 4; ++i) {
      jets[i].a = parameters[0][i];
      jets[i].v.setZero();
      jets[i].v[i] = 1.0;
    }

    ceres::Jet<double, 4> result;
    (*functor_)(jets, &result);

    // Copy the values out of the Jet.
    residuals[0] = result.a;
    for (int i = 0; i < 4; ++i) {
      jacobians[0][i] = result.v[i];
      cout << "jet" << endl;
      cout << jacobians[0][i] << endl;
    }
    return true;
  }

 private:
  std::unique_ptr<const Rat43CostFunctor> functor_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  double initial_b1 = 5.0;
  double b1 = initial_b1;

  double initial_b2 = 3.0;
  double b2 = initial_b2;

  double initial_b3 = 2.0;
  double b3 = initial_b3;

  double initial_b4 = 4.0;
  double b4 = initial_b4;

  vector<double> param = {b1, b2, b3, b4};

  Problem problem;
  for (int i = 0; i < kNumObservations; ++i) {
    problem.AddResidualBlock(
        new AutoDiffCostFunction<Rat43CostFunctor, 1, 4>(
          new Rat43CostFunctor(data[2 * i], data[2 * i + 1])),
        nullptr,
        param.data());
  }

  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR; 
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";

  std::cout << "b1 : " << initial_b1
            << " -> " << b1 << "\n"
            << "b2 : " << initial_b2
            << " -> " << b2 << "\n"
            << "b3 : " << initial_b3
            << " -> " << b3 << "\n"
            << "b4 : " << initial_b4
            << " -> " << b4 << "\n";

  return 0;
}
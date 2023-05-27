#pragma once

template <int dim>
class GaussianSolution : public dealii::Function<dim>
{
public:
  GaussianSolution(const std::vector<Point<dim>> &source_centers,
                   const double                   width)
    : dealii::Function<dim>()
    , source_centers(source_centers)
    , width(width)
  {}

  double
  value(dealii::Point<dim> const &p,
        unsigned int const /*component*/ = 0) const override
  {
    double return_value = 0;

    for (const auto &source_center : this->source_centers)
      {
        const dealii::Tensor<1, dim> x_minus_xi = p - source_center;
        return_value +=
          std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
      }

    return return_value / dealii::Utilities::fixed_power<dim>(
                            std::sqrt(2. * dealii::numbers::PI) * this->width);
  }

private:
  const std::vector<Point<dim>> source_centers;
  const double                  width;
};

template <int dim>
class GaussianRightHandSide : public dealii::Function<dim>
{
public:
  GaussianRightHandSide(const std::vector<Point<dim>> &source_centers,
                        const double                   width)
    : dealii::Function<dim>()
    , source_centers(source_centers)
    , width(width)
  {}

  double
  value(dealii::Point<dim> const &p,
        unsigned int const /*component*/ = 0) const override
  {
    double const coef         = 1.0;
    double       return_value = 0;

    for (const auto &source_center : this->source_centers)
      {
        const dealii::Tensor<1, dim> x_minus_xi = p - source_center;

        return_value +=
          ((2 * dim * coef -
            4 * coef * x_minus_xi.norm_square() / (this->width * this->width)) /
           (this->width * this->width) *
           std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
      }

    return return_value / dealii::Utilities::fixed_power<dim>(
                            std::sqrt(2 * dealii::numbers::PI) * this->width);
  }

private:
  const std::vector<Point<dim>> source_centers;
  const double                  width;
};

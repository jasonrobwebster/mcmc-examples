import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--intercept', default=1, type=float, help='true intercept of linear function')
parser.add_argument('--gradient', default=2, type=float, help='true gradient of linear function')
parser.add_argument('-m', '--mu', default=0, type=float, help='mean of additive noise')
parser.add_argument('-s', '--sigma', default=0.5, type=float, help='standard deviation of additive noise')
parser.add_argument('--samples', default=10000, type=int)


def create_data(x, intercept: float, gradient: float, mu: float, sigma: float):
    x = np.asarray(x)
    y = intercept + gradient * x + np.random.normal(loc=mu, scale=sigma, size=x.size)
    return y


if __name__ == '__main__':
    args = parser.parse_args()
    true_intercept, true_gradient, mu_data, sigma_data = args.intercept, args.gradient, args.mu, args.sigma

    x = np.linspace(0, 10, 100)
    data = create_data(x, true_intercept, true_gradient, mu_data, sigma_data)

    with pm.Model() as poly_model:
        intercept = pm.Uniform('Intercept', -10, 10)
        x_coeff = pm.Uniform('x', -10, 10)
        sigma = pm.Uniform('sigma', 0, 20)

        y_obs = pm.Normal('y_obs', mu=intercept + x_coeff * x, sigma=sigma, observed=data)

        trace = pm.sample(args.samples, cores=4)

    print(pm.summary(trace).round(2))
    pm.traceplot(trace)
    plt.show()

    plt.plot(x, data, 'x')
    pm.plot_posterior_predictive_glm(trace, eval=x)
    plt.show()

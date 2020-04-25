import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-a', default=3, type=float, help='quadratic coefficient')
parser.add_argument('-b', default=2, type=float, help='linear coefficient')
parser.add_argument('-c', default=1, type=float, help='constant')
parser.add_argument('-m', '--mu', default=0, type=float, help='mean of additive noise')
parser.add_argument('-s', '--sigma', default=0.5, type=float, help='standard deviation of additive noise')
parser.add_argument('--samples', default=10000, type=int)


def create_data(x, a: float, b: float, c: float, mu: float, sigma: float):
    x = np.asarray(x)
    y = a * x**2 + b * x + c + np.random.normal(loc=mu, scale=sigma, size=x.size)
    return y


if __name__ == '__main__':
    args = parser.parse_args()
    true_a, true_b, true_c, mu_data, sigma_data = args.a, args.b, args.c, args.mu, args.sigma

    x = np.linspace(0, 1, 1000)
    data = create_data(x, true_a, true_b, true_c, mu_data, sigma_data)

    with pm.Model() as poly_model:
        a = pm.Uniform('a', -10, 10)
        b = pm.Uniform('b', -10, 10)
        c = pm.Uniform('c', -10, 10)
        sigma = pm.Uniform('sigma', 0, 20)

        y_obs = pm.Normal('y_obs', mu=a * x**2 + b * x + c, sigma=sigma, observed=data)

        trace = pm.sample(args.samples, cores=4)

    print(pm.summary(trace, credible_interval=0.95).round(2))
    # pm.plot_posterior(trace)
    # pm.pairplot(trace)
    plt.show()

    plt.plot(x, data, 'x')
    pm.plot_posterior_predictive_glm(trace, lm=lambda x, sample: sample['a'] * x**2 + sample['b'] * x + sample['c'], eval=x)
    plt.show()

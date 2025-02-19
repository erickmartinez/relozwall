import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt
from scipy.special import gamma


class ThompsonDistribution:
    def __init__(self, n, lambda_):
        self.n = n
        self.lambda_ = lambda_

    def pdf(self, x):
        """Probability density function of Thompson energy distribution"""
        const = (self.n * (self.lambda_ ** self.n)) / gamma(self.n)
        return const * (x ** (self.n - 1)) * np.exp(-self.lambda_ * x)

    def moment(self, k):
        """Calculate the k-th moment of the distribution"""
        return (gamma(self.n + k) / (self.lambda_ ** k)) / gamma(self.n)

    def mean(self):
        """Calculate mean of the distribution"""
        return self.moment(1)

    def variance(self):
        """Calculate variance of the distribution"""
        return self.moment(2) - self.moment(1) ** 2

    def skewness(self):
        """Calculate skewness of the distribution"""
        m1 = self.moment(1)
        m2 = self.moment(2)
        m3 = self.moment(3)
        var = m2 - m1 ** 2
        return (m3 - 3 * m1 * var - m1 ** 3) / (var ** (3 / 2))

    def kurtosis(self):
        """Calculate kurtosis of the distribution"""
        m1 = self.moment(1)
        m2 = self.moment(2)
        m3 = self.moment(3)
        m4 = self.moment(4)
        var = m2 - m1 ** 2
        return (m4 - 4 * m1 * m3 + 6 * (m1 ** 2) * m2 - 3 * m1 ** 4) / (var ** 2)

    def error_M(self, M):
        """Calculate M-th error of the distribution"""
        return np.sqrt(self.moment(2 * M) - self.moment(M) ** 2)

    def mode(self):
        """Calculate mode of the distribution"""
        return (self.n - 1) / self.lambda_


def fit_thompson(target_stats):
    """
    Fit Thompson distribution parameters to match target statistics

    Parameters:
    target_stats: dict containing target statistics
    """

    def objective(params):
        n, lambda_ = params
        if n <= 0 or lambda_ <= 0:
            return np.inf

        dist = ThompsonDistribution(n, lambda_)

        errors = []
        # Match mean
        errors.append(((dist.mean() - target_stats['mean']) / target_stats['mean']) ** 2)
        # Match variance
        errors.append(((dist.variance() - target_stats['variance']) / target_stats['variance']) ** 2)
        # Match skewness
        errors.append(((dist.skewness() - target_stats['skewness']) / target_stats['skewness']) ** 2)
        # Match kurtosis
        errors.append(((dist.kurtosis() - target_stats['kurtosis']) / target_stats['kurtosis']) ** 2)

        return np.sum(errors)

    # Initial guess for parameters
    initial_guess = [2.0, 1.0]

    # Optimize parameters
    result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')

    return ThompsonDistribution(result.x[0], result.x[1])


# Example usage with sample target statistics
target_stats = {
    'mean': 3.572500,
    'variance': 5.010100,
    'skewness': 8.361500e-01,
    'kurtosis': 3.407900,
}

# Fit distribution
fitted_dist = fit_thompson(target_stats)

# Compare original and fitted statistics
print("Comparison of Statistics:")
print(f"{'Statistic':<15} {'Original':<15} {'Fitted':<15} {'Relative Error (%)':<15}")
print("-" * 60)

stats_to_compare = {
    'Mean': (target_stats['mean'], fitted_dist.mean()),
    'Variance': (target_stats['variance'], fitted_dist.variance()),
    'Skewness': (target_stats['skewness'], fitted_dist.skewness()),
    'Kurtosis': (target_stats['kurtosis'], fitted_dist.kurtosis())
}

for stat_name, (orig, fitted) in stats_to_compare.items():
    rel_error = abs((fitted - orig) / orig * 100)
    print(f"{stat_name:<15} {orig:<15.4f} {fitted:<15.4f} {rel_error:<15.4f}")

# Print moments 1 to 6
print("\nMoments 1 to 6:")
for i in range(1, 7):
    moment = fitted_dist.moment(i)
    print(f"Moment {i}: {moment:.4f}")

# Print errors 1.M to 3.M
print("\nErrors 1.M to 3.M:")
for M in range(1, 4):
    error = fitted_dist.error_M(M)
    print(f"Error {M}.M: {error:.4f}")

# Calculate and print the mode
mode = fitted_dist.mode()
print(f"\nMode of fitted distribution: {mode:.4f}")

# Plot the fitted distribution
x = np.linspace(0, 10, 1000)
y = fitted_dist.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='Fitted Thompson Distribution')
plt.axvline(x=mode, color='r', linestyle='--', label='Mode')
plt.axvline(x=fitted_dist.mean(), color='g', linestyle='--', label='Mean')
plt.title('Fitted Thompson Energy Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
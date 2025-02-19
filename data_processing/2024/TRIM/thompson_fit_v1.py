import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt


class ThompsonSputteringDistribution:
    def __init__(self, Eb, M):
        """
        Initialize Thompson distribution for sputtering with known Eb

        Parameters:
        Eb : float
            Surface binding energy (eV) - known material property
        M : float
            Power law parameter for high energy tail
        """
        self.Eb = Eb
        self.M = M

    def pdf(self, E):
        """
        Probability density function of Thompson energy distribution for sputtering

        Parameters:
        E : array_like
            Energy values (eV)
        """
        mask = E >= self.Eb
        result = np.zeros_like(E, dtype=float)

        # Thompson distribution formula for sputtering
        result[mask] = (self.Eb * E[mask] / (E[mask] + self.Eb) ** 3) * (1 / self.M)

        # Normalize the distribution
        norm_const = np.trapz(result, E)
        if norm_const > 0:
            result /= norm_const

        return result

    def moment(self, k):
        """Calculate the k-th moment of the distribution"""
        E_max = 1000 * self.Eb  # Upper limit for integration
        E = np.linspace(self.Eb, E_max, 10000)
        integrand = self.pdf(E) * (E ** k)
        return np.trapz(integrand, E)

    def mean(self):
        return self.moment(1)

    def variance(self):
        return self.moment(2) - self.moment(1) ** 2

    def skewness(self):
        m1 = self.moment(1)
        m2 = self.moment(2)
        m3 = self.moment(3)
        var = m2 - m1 ** 2
        return (m3 - 3 * m1 * var - m1 ** 3) / (var ** (3 / 2))

    def kurtosis(self):
        m1 = self.moment(1)
        m2 = self.moment(2)
        m3 = self.moment(3)
        m4 = self.moment(4)
        var = m2 - m1 ** 2
        return (m4 - 4 * m1 * m3 + 6 * (m1 ** 2) * m2 - 3 * m1 ** 4) / (var ** 2)

    def error_M(self, k):
        return np.sqrt(self.moment(2 * k) - self.moment(k) ** 2)

    def mode(self):
        return self.Eb / 2


def fit_thompson_sputtering(target_stats, Eb, method='de'):
    """
    Fit Thompson sputtering distribution with known Eb

    Parameters:
    target_stats : dict
        Dictionary containing target statistics
    Eb : float
        Known surface binding energy (eV)
    method : str
        'de' for differential evolution or 'nm' for Nelder-Mead
    """

    def objective(M):
        if M <= 1:  # M > 1 required for convergence
            return np.inf

        dist = ThompsonSputteringDistribution(Eb, M[0])

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

    if method == 'de':
        # Differential Evolution with physics-informed bounds for M
        bounds = [(1.1, 3.0)]  # bounds for M only
        result = optimize.differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=1e-7,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=True
        )
    else:
        # Nelder-Mead with physics-informed initial guess
        initial_guess = [2.0]  # Initial guess for M
        result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')

    return ThompsonSputteringDistribution(Eb, result.x[0])


# Example usage with realistic sputtering statistics and known Eb
Eb_known = 5.73 # Example: known binding energy in eV

target_stats = {
    'mean': 3.572500,
    'variance': 5.010100,
    'skewness': 8.361500e-01,
    'kurtosis': 3.407900,
}

# Fit distribution using both methods
fitted_dist_de = fit_thompson_sputtering(target_stats, Eb_known, method='de')
fitted_dist_nm = fit_thompson_sputtering(target_stats, Eb_known, method='nm')


# Function to print comparison statistics
def print_comparison_stats(dist, target_stats, method_name):
    print(f"\nComparison of Statistics ({method_name}):")
    print(f"{'Statistic':<15} {'Original':<15} {'Fitted':<15} {'Relative Error (%)':<15}")
    print("-" * 60)

    stats_to_compare = {
        'Mean': (target_stats['mean'], dist.mean()),
        'Variance': (target_stats['variance'], dist.variance()),
        'Skewness': (target_stats['skewness'], dist.skewness()),
        'Kurtosis': (target_stats['kurtosis'], dist.kurtosis())
    }

    for stat_name, (orig, fitted) in stats_to_compare.items():
        rel_error = abs((fitted - orig) / orig * 100)
        print(f"{stat_name:<15} {orig:<15.4f} {fitted:<15.4f} {rel_error:<15.4f}")

    print(f"\nMoments 1 to 6 ({method_name}):")
    for i in range(1, 7):
        moment = dist.moment(i)
        print(f"Moment {i}: {moment:.4f}")

    print(f"\nErrors 1.M to 3.M ({method_name}):")
    for M in range(1, 4):
        error = dist.error_M(M)
        print(f"Error {M}.M: {error:.4f}")

    mode = dist.mode()
    print(f"\nMode of fitted distribution ({method_name}): {mode:.4f}")
    return mode


# Print comparisons for both methods
mode_de = print_comparison_stats(fitted_dist_de, target_stats, "Differential Evolution")
mode_nm = print_comparison_stats(fitted_dist_nm, target_stats, "Nelder-Mead")

# Plot both fitted distributions
E = np.linspace(0, 30, 1000)
y_de = fitted_dist_de.pdf(E)
y_nm = fitted_dist_nm.pdf(E)

plt.figure(figsize=(12, 7))
plt.plot(E, y_de, 'b-', label='DE Fitted Distribution')
plt.plot(E, y_nm, 'r--', label='NM Fitted Distribution')
plt.axvline(x=mode_de, color='blue', linestyle=':', label=f'Mode (Eb/2 = {mode_de:.2f} eV)')
plt.axvline(x=fitted_dist_de.mean(), color='green', linestyle='--', label=f'Mean ({fitted_dist_de.mean():.2f} eV)')

# Add E^(-2) reference line
E_ref = np.linspace(10, 30, 100)
ref_line = 0.1 * E_ref ** (-2)  # Scale factor added for visualization
plt.plot(E_ref, ref_line, 'k:', label='E^(-2) reference')

plt.title(f'Thompson Energy Distribution for Sputtering (Eb = {Eb_known:.1f} eV)')
plt.xlabel('Energy (eV)')
plt.ylabel('Probability Density')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Print optimization parameters
print("\nOptimization Parameters:")
print(f"DE: Eb = {Eb_known:.6f} eV (fixed), M = {fitted_dist_de.M:.6f}")
print(f"NM: Eb = {Eb_known:.6f} eV (fixed), M = {fitted_dist_nm.M:.6f}")
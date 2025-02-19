import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import warnings


class SkewNormalFitter:
    def __init__(self, target_stats):
        """
        Initialize the fitter with target statistics.

        Parameters:
        target_stats (dict): Dictionary containing target statistics:
            - mean: Mean of the distribution
            - variance: Variance of the distribution
            - skewness: Skewness of the distribution
            - kurtosis: Kurtosis of the distribution
            - error_1M to error_3M: Error metrics
            - moments_1 to moments_6: First 6 moments
        """
        self.target_stats = target_stats
        self.weights = {
            'mean': 1.0,
            'variance': 1.0,
            'skewness': 1.0,
            'kurtosis': 1.0,
            'moments': [1.0] * 6,
            'errors': [1.0] * 3
        }

    def calculate_moments(self, a, loc, scale):
        """Calculate the first 6 moments of the skew normal distribution."""
        moments = []
        for i in range(1, 7):
            moment = skewnorm.moment(i, a, loc=loc, scale=scale)
            moments.append(moment)
        return moments

    def objective_function(self, params):
        """
        Objective function to minimize the difference between target and fitted statistics.

        Parameters:
        params (array-like): [a (shape), loc (location), scale]
        """
        a, loc, scale = params

        # Calculate statistics for current parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                current_mean = skewnorm.mean(a, loc=loc, scale=scale)
                current_var = skewnorm.var(a, loc=loc, scale=scale)
                current_skew = skewnorm.stats_ono(a, loc=loc, scale=scale, moments='s')
                current_kurt = skewnorm.stats_ono(a, loc=loc, scale=scale, moments='k')
                current_moments = self.calculate_moments(a, loc, scale)

                # Calculate error metrics (simplified example)
                current_errors = [
                    abs(current_mean - self.target_stats['mean']),
                    abs(current_var - self.target_stats['variance']),
                    abs(current_skew - self.target_stats['skewness'])
                ]

                # Calculate total error
                error = (
                        self.weights['mean'] * (current_mean - self.target_stats['mean']) ** 2 +
                        self.weights['variance'] * (current_var - self.target_stats['variance']) ** 2 +
                        self.weights['skewness'] * (current_skew - self.target_stats['skewness']) ** 2 +
                        self.weights['kurtosis'] * (current_kurt - self.target_stats['kurtosis']) ** 2
                )

                # Add moment errors
                for i in range(6):
                    error += self.weights['moments'][i] * (
                            current_moments[i] - self.target_stats[f'moment_{i + 1}']
                    ) ** 2

                # Add error metrics
                for i in range(3):
                    error += self.weights['errors'][i] * (
                            current_errors[i] - self.target_stats[f'error_{i + 1}M']
                    ) ** 2

                return error

            except (ValueError, RuntimeWarning):
                return float('inf')

    def plot_fit(self, params, fitted_stats):
        """
        Plot the fitted distribution and compare statistics.

        Parameters:
        params (array-like): Fitted parameters [a, loc, scale]
        fitted_stats (dict): Dictionary containing fitted statistics
        """
        a, loc, scale = params

        # Generate points for plotting
        x = np.linspace(loc - 4 * scale, loc + 4 * scale, 1000)
        y = skewnorm.pdf(x, a, loc=loc, scale=scale)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot distribution
        ax1.plot(x, y, 'b-', lw=2, label='Fitted Distribution')
        ax1.set_title('Fitted Skew Normal Distribution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability Density')
        ax1.grid(True)
        ax1.legend()

        # Prepare comparison data
        stats_comparison = {
            'Statistic': [
                             'Mean', 'Variance', 'Skewness', 'Kurtosis',
                             'Error 1M', 'Error 2M', 'Error 3M'
                         ] + [f'Moment {i}' for i in range(1, 7)],
            'Target': [
                          self.target_stats['mean'],
                          self.target_stats['variance'],
                          self.target_stats['skewness'],
                          self.target_stats['kurtosis'],
                          self.target_stats['error_1M'],
                          self.target_stats['error_2M'],
                          self.target_stats['error_3M']
                      ] + [self.target_stats[f'moment_{i}'] for i in range(1, 7)],
            'Fitted': [
                          fitted_stats['statistics']['mean'],
                          fitted_stats['statistics']['variance'],
                          fitted_stats['statistics']['skewness'],
                          fitted_stats['statistics']['kurtosis'],
                          # Calculate errors for fitted distribution
                          abs(fitted_stats['statistics']['mean'] - self.target_stats['mean']),
                          abs(fitted_stats['statistics']['variance'] - self.target_stats['variance']),
                          abs(fitted_stats['statistics']['skewness'] - self.target_stats['skewness'])
                      ] + fitted_stats['statistics']['moments'],
        }

        # Calculate relative differences
        stats_comparison['Rel. Diff (%)'] = [
            abs(f - t) / abs(t) * 100 if abs(t) > 1e-10 else abs(f - t) * 100
            for f, t in zip(stats_comparison['Fitted'], stats_comparison['Target'])
        ]

        # Create comparison table
        df = pd.DataFrame(stats_comparison)
        df = df.round(4)

        # Plot table
        ax2.axis('off')
        table = ax2.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.tight_layout()
        plt.show()

        return df

    def fit(self, initial_guess=None, plot=True):
        """
        Fit the skew normal distribution to the target statistics.

        Parameters:
        initial_guess (array-like, optional): Initial guess for [a, loc, scale]
        plot (bool): Whether to plot the results

        Returns:
        tuple: (a, loc, scale) - Fitted parameters
        dict: Fitting results and statistics
        pandas.DataFrame: Comparison of target and fitted statistics (if plot=True)
        """
        if initial_guess is None:
            initial_guess = [1.0, 0.0, 1.0]

        # Set bounds for parameters
        bounds = [
            (-10, 10),  # a (shape)
            (-10, 10),  # loc (location)
            (0.001, 10)  # scale (must be positive)
        ]

        # Perform optimization
        result = minimize(
            self.objective_function,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )

        # Get fitted parameters
        a_fit, loc_fit, scale_fit = result.x

        # Calculate final statistics
        fitted_stats = {
            'parameters': {
                'a': a_fit,
                'loc': loc_fit,
                'scale': scale_fit
            },
            'convergence': result.success,
            'error': result.fun,
            'statistics': {
                'mean': skewnorm.mean(a_fit, loc=loc_fit, scale=scale_fit),
                'variance': skewnorm.var(a_fit, loc=loc_fit, scale=scale_fit),
                'skewness': skewnorm.stats_ono(a_fit, loc=loc_fit, scale=scale_fit, moments='s'),
                'kurtosis': skewnorm.stats_ono(a_fit, loc=loc_fit, scale=scale_fit, moments='k'),
                'moments': self.calculate_moments(a_fit, loc_fit, scale_fit)
            }
        }

        # Plot results if requested
        comparison_df = None
        if plot:
            comparison_df = self.plot_fit(result.x, fitted_stats)

        return result.x, fitted_stats, comparison_df


# Example usage
if __name__ == "__main__":
    # Example target statistics
    target_stats = {
        'mean': 3.572500,
        'variance': 5.010100,
        'skewness': 8.361500e-01,
        'kurtosis': 3.407900,
        'error_1M': 7.246800e-02,
        'error_2M': 2.517000e-01,
        'error_3M': 7.817900e-02,
        'moment_1': 3.572500,
        'moment_2': 5.010100,
        'moment_3': 8.361500e-01,
        'moment_4': 3.407900e+00,
        'moment_5': 2.238300e+00,
        'moment_6': 7.246800e-02
    }

    # Create fitter instance
    fitter = SkewNormalFitter(target_stats)

    # Fit distribution
    params, results, comparison = fitter.fit(plot=True)

    print("\nFitted parameters:")
    print(f"Shape (a): {params[0]:.4f}")
    print(f"Location (loc): {params[1]:.4f}")
    print(f"Scale: {params[2]:.4f}")

    print("\nOptimization success:", results['convergence'])
    print("Final error:", results['error'])
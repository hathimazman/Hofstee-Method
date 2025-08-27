import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import fsolve, brentq
import warnings
warnings.filterwarnings('ignore')

class AnalyticalHofsteeAnalyzer:
    def __init__(self, scores, min_cutoff=48, max_cutoff=50, min_fail_rate=0.01, max_fail_rate=0.07):
        """
        Initialize Hofstee analyzer with analytical equation solving
        """
        self.scores = np.array(scores, dtype=float)
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff
        self.min_fail_rate = min_fail_rate
        self.max_fail_rate = max_fail_rate
        
        print(f"Loaded {len(self.scores)} scores")
        print(f"Score range: {np.min(self.scores):.2f} - {np.max(self.scores):.2f}")
        print(f"Unique scores: {len(np.unique(self.scores))}")
        
        # Create the interpolation function and diagonal equation
        self.spline_func = None
        self.diagonal_func = None
        self._setup_equations()
    
    def create_empirical_cdf_points(self):
        """Create empirical cumulative distribution points"""
        sorted_scores = np.sort(self.scores)
        n_total = len(sorted_scores)
        
        unique_scores, counts = np.unique(sorted_scores, return_counts=True)
        cumulative_counts = np.cumsum(counts)
        
        empirical_scores = []
        empirical_percentages = []
        
        # Add starting point
        empirical_scores.append(np.min(sorted_scores) - 0.1)
        empirical_percentages.append(0.0)
        
        # Add points for each unique score
        for i, score in enumerate(unique_scores):
            if i == 0:
                below_count = 0
            else:
                below_count = cumulative_counts[i-1]
            
            percentage_below = (below_count / n_total) * 100
            empirical_scores.append(score)
            empirical_percentages.append(percentage_below)
        
        # Add ending point
        empirical_scores.append(np.max(sorted_scores) + 0.1)
        empirical_percentages.append(100.0)
        
        return np.array(empirical_scores), np.array(empirical_percentages)
    
    def _setup_equations(self):
        """Set up the interpolation function and diagonal equation"""
        # Get empirical CDF points
        empirical_scores, empirical_percentages = self.create_empirical_cdf_points()
        
        # Create cubic spline interpolation function
        self.spline_func = CubicSpline(empirical_scores, empirical_percentages, bc_type='natural')
        
        # Define diagonal function: y = mx + b
        # The diagonal goes from (min_cutoff, max_fail_rate*100) to (max_cutoff, min_fail_rate*100)
        slope = (self.min_fail_rate * 100 - self.max_fail_rate * 100) / (self.max_cutoff - self.min_cutoff)
        intercept = self.max_fail_rate * 100 - slope * self.min_cutoff
        
        self.diagonal_func = lambda x: slope * x + intercept
        
        print(f"Diagonal equation: y = {slope:.6f}x + {intercept:.6f}")
        
        return slope, intercept
    
    def intersection_equation(self, x):
        """
        Function to find intersection: spline(x) - diagonal(x) = 0
        """
        return self.spline_func(x) - self.diagonal_func(x)
    
    def solve_hofstee_analytical(self, method='brentq'):
        """
        Solve for Hofstee cutoff analytically by finding where:
        spline_function(x) = diagonal_function(x)
        """
        try:
            if method == 'brentq':
                # Brent's method - very reliable for finding roots
                cutoff_score = brentq(self.intersection_equation, self.min_cutoff, self.max_cutoff)
            
            elif method == 'fsolve':
                # Use initial guess at midpoint
                initial_guess = (self.min_cutoff + self.max_cutoff) / 2
                cutoff_score = fsolve(self.intersection_equation, initial_guess)[0]
            
            else:  # 'newton' or other scipy methods
                from scipy.optimize import newton
                initial_guess = (self.min_cutoff + self.max_cutoff) / 2
                cutoff_score = newton(self.intersection_equation, initial_guess)
            
            # Calculate corresponding failure rate
            failure_rate_percentage = self.spline_func(cutoff_score)
            failure_rate = failure_rate_percentage / 100
            
            # Verify with diagonal
            diagonal_value = self.diagonal_func(cutoff_score)
            intersection_error = abs(failure_rate_percentage - diagonal_value)
            
            return {
                'cutoff': cutoff_score,
                'failure_rate': failure_rate,
                'failure_rate_percentage': failure_rate_percentage,
                'diagonal_value': diagonal_value,
                'intersection_error': intersection_error,
                'method': method,
                'analytical_solution': True
            }
            
        except Exception as e:
            print(f"Analytical solution failed: {e}")
            # Fallback to numerical approximation
            return self._fallback_numerical_solution()
    
    def _fallback_numerical_solution(self):
        """Fallback numerical solution if analytical fails"""
        score_range = np.linspace(self.min_cutoff, self.max_cutoff, 10000)
        spline_values = self.spline_func(score_range)
        diagonal_values = [self.diagonal_func(x) for x in score_range]
        
        differences = np.abs(spline_values - diagonal_values)
        min_idx = np.argmin(differences)
        
        cutoff_score = score_range[min_idx]
        failure_rate_percentage = spline_values[min_idx]
        
        return {
            'cutoff': cutoff_score,
            'failure_rate': failure_rate_percentage / 100,
            'failure_rate_percentage': failure_rate_percentage,
            'diagonal_value': diagonal_values[min_idx],
            'intersection_error': differences[min_idx],
            'method': 'numerical_fallback',
            'analytical_solution': False
        }
    
    def get_spline_coefficients(self, x_point):
        """
        Get the cubic spline coefficients for a specific segment containing x_point
        Returns the polynomial coefficients for that segment
        """
        # Find which segment contains x_point
        empirical_scores, _ = self.create_empirical_cdf_points()
        
        # Find the segment
        segment_idx = np.searchsorted(empirical_scores, x_point) - 1
        segment_idx = max(0, min(segment_idx, len(empirical_scores) - 2))
        
        # Get the spline coefficients for this segment
        # CubicSpline stores coefficients internally
        if hasattr(self.spline_func, 'c'):
            coeffs = self.spline_func.c[:, segment_idx]  # [c3, c2, c1, c0] for ax^3 + bx^2 + cx + d
            x_start = empirical_scores[segment_idx]
            
            return {
                'coefficients': coeffs,
                'segment_start': x_start,
                'segment_idx': segment_idx,
                'polynomial': f"{coeffs[0]:.6f}(x-{x_start:.2f})³ + {coeffs[1]:.6f}(x-{x_start:.2f})² + {coeffs[2]:.6f}(x-{x_start:.2f}) + {coeffs[3]:.6f}"
            }
        else:
            return None
    
    def compare_solution_methods(self):
        """Compare different analytical solution methods"""
        methods = ['brentq', 'fsolve']
        results = []
        
        for method in methods:
            try:
                result = self.solve_hofstee_analytical(method=method)
                results.append({
                    'Method': method.title(),
                    'Cutoff': f"{result['cutoff']:.6f}",
                    'Failure Rate': f"{result['failure_rate']:.4%}",
                    'Intersection Error': f"{result['intersection_error']:.2e}",
                    'Status': 'Success' if result['analytical_solution'] else 'Fallback'
                })
            except Exception as e:
                results.append({
                    'Method': method.title(),
                    'Cutoff': 'Failed',
                    'Failure Rate': 'Failed',
                    'Intersection Error': 'Failed',
                    'Status': f'Error: {str(e)[:30]}'
                })
        
        return pd.DataFrame(results)
    
    def plot_analytical_solution(self, figsize=(14, 10)):
        """Create comprehensive plot showing analytical solution"""
        # Solve analytically
        results = self.solve_hofstee_analytical(method='brentq')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Analytical Hofstee Method: Equation-Based Solution', fontsize=16, fontweight='bold')
        
        # Plot 1: Full cumulative curve with intersection
        full_range = np.linspace(np.min(self.scores) - 2, np.max(self.scores) + 2, 1000)
        spline_values = self.spline_func(full_range)
        
        ax1.plot(full_range, spline_values, 'b-', linewidth=3, label='Cubic Spline Function')
        
        # Diagonal line (extended for visualization)
        diagonal_extended = [self.diagonal_func(x) for x in full_range]
        ax1.plot(full_range, diagonal_extended, 'g--', linewidth=3, alpha=0.7, label='Diagonal Function')
        
        # Mark intersection
        ax1.plot(results['cutoff'], results['failure_rate_percentage'], 'ro', 
                markersize=15, markeredgecolor='black', markeredgewidth=3, zorder=10)
        
        # Constraint box
        rect_x = [self.min_cutoff, self.max_cutoff, self.max_cutoff, self.min_cutoff, self.min_cutoff]
        rect_y = [self.min_fail_rate * 100, self.min_fail_rate * 100, 
                  self.max_fail_rate * 100, self.max_fail_rate * 100, self.min_fail_rate * 100]
        ax1.plot(rect_x, rect_y, 'r-', linewidth=2, label='Hofstee Constraints')
        ax1.fill(rect_x[:-1], rect_y[:-1], alpha=0.1, color='red')
        
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Cumulative Percentage')
        ax1.set_title('Analytical Solution: Function Intersection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(np.min(self.scores) - 2, np.max(self.scores) + 2)
        ax1.set_ylim(-2, 102)
        
        # Add precise annotation
        ax1.annotate(f'Analytical Solution\nCutoff: {results["cutoff"]:.4f}\nFailure Rate: {results["failure_rate"]:.3%}\nError: {results["intersection_error"]:.2e}',
                    xy=(results['cutoff'], results['failure_rate_percentage']), 
                    xytext=(results['cutoff'] + 5, results['failure_rate_percentage'] + 15),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
                    fontsize=10, fontweight='bold')
        
        # Plot 2: Zoomed intersection region
        zoom_range = np.linspace(results['cutoff'] - 2, results['cutoff'] + 2, 500)
        zoom_spline = self.spline_func(zoom_range)
        zoom_diagonal = [self.diagonal_func(x) for x in zoom_range]
        
        ax2.plot(zoom_range, zoom_spline, 'b-', linewidth=3, label='Spline Function')
        ax2.plot(zoom_range, zoom_diagonal, 'g--', linewidth=3, label='Diagonal Function')
        ax2.plot(results['cutoff'], results['failure_rate_percentage'], 'ro', 
                markersize=12, markeredgecolor='black', markeredgewidth=2)
        
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Cumulative Percentage')
        ax2.set_title('Intersection Detail (Zoomed)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Difference function (should cross zero at intersection)
        diff_range = np.linspace(self.min_cutoff - 1, self.max_cutoff + 1, 1000)
        difference_values = [self.intersection_equation(x) for x in diff_range]
        
        ax3.plot(diff_range, difference_values, 'purple', linewidth=3, label='Spline - Diagonal')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='y = 0')
        ax3.axvline(x=results['cutoff'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Solution: x = {results["cutoff"]:.4f}')
        ax3.plot(results['cutoff'], 0, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
        
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Difference (Spline - Diagonal)')
        ax3.set_title('Difference Function (Root = Intersection)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Method comparison
        comparison_df = self.compare_solution_methods()
        
        # Create a simple table visualization
        ax4.axis('tight')
        ax4.axis('off')
        table_data = []
        for _, row in comparison_df.iterrows():
            table_data.append([row['Method'], row['Cutoff'], row['Failure Rate'], row['Status']])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Method', 'Cutoff', 'Failure Rate', 'Status'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Method Comparison', fontweight='bold')
        
        plt.tight_layout()
        return fig, results
    
    def get_equation_details(self):
        """Get detailed information about the equations used"""
        # Get spline coefficient info at the intersection
        results = self.solve_hofstee_analytical()
        spline_info = self.get_spline_coefficients(results['cutoff'])
        
        # Diagonal equation parameters
        slope = (self.min_fail_rate * 100 - self.max_fail_rate * 100) / (self.max_cutoff - self.min_cutoff)
        intercept = self.max_fail_rate * 100 - slope * self.min_cutoff
        
        details = {
            'diagonal_equation': f"y = {slope:.6f}x + {intercept:.6f}",
            'diagonal_slope': slope,
            'diagonal_intercept': intercept,
            'spline_segment_info': spline_info,
            'intersection_point': (results['cutoff'], results['failure_rate_percentage']),
            'intersection_error': results['intersection_error'],
            'solution_method': results['method']
        }
        
        return details
    
    def print_detailed_analysis(self):
        """Print comprehensive analysis with equation details"""
        print("="*60)
        print("ANALYTICAL HOFSTEE ANALYSIS")
        print("="*60)
        
        # Basic info
        print(f"Dataset: {len(self.scores)} scores")
        print(f"Range: {np.min(self.scores):.2f} - {np.max(self.scores):.2f}")
        print(f"Constraints: {self.min_cutoff} ≤ cutoff ≤ {self.max_cutoff}")
        print(f"Failure rates: {self.min_fail_rate:.1%} ≤ rate ≤ {self.max_fail_rate:.1%}")
        print()
        
        # Equation details
        details = self.get_equation_details()
        print("MATHEMATICAL EQUATIONS:")
        print("-" * 30)
        print(f"Diagonal function: {details['diagonal_equation']}")
        print(f"Spline function: Piecewise cubic polynomial")
        if details['spline_segment_info']:
            print(f"At intersection segment: {details['spline_segment_info']['polynomial']}")
        print()
        
        # Solution
        results = self.solve_hofstee_analytical()
        print("ANALYTICAL SOLUTION:")
        print("-" * 30)
        print(f"Method used: {results['method']}")
        print(f"Hofstee cutoff: {results['cutoff']:.6f}")
        print(f"Failure rate: {results['failure_rate']:.4%}")
        print(f"Intersection error: {results['intersection_error']:.2e}")
        print(f"Analytical solution: {'Yes' if results['analytical_solution'] else 'No (fallback used)'}")
        print()
        
        # Verification
        print("VERIFICATION:")
        print("-" * 30)
        spline_value = self.spline_func(results['cutoff'])
        diagonal_value = self.diagonal_func(results['cutoff'])
        print(f"Spline value at cutoff: {spline_value:.6f}")
        print(f"Diagonal value at cutoff: {diagonal_value:.6f}")
        print(f"Absolute difference: {abs(spline_value - diagonal_value):.2e}")
        print()
        
        # Pass/fail statistics
        passing = np.sum(self.scores >= results['cutoff'])
        failing = len(self.scores) - passing
        print("OUTCOME STATISTICS:")
        print("-" * 30)
        print(f"Students passing: {passing} ({passing/len(self.scores):.1%})")
        print(f"Students failing: {failing} ({failing/len(self.scores):.1%})")
        
        print("="*60)

# Example usage with your data
def demo_analytical_hofstee():
    """Demonstrate analytical Hofstee method"""
    # Your actual data
    scores_data = [
        43.09, 43.44, 43.57, 46.03, 49.33, 49.85, 49.88, 49.99, 50.11, 50.21,
        50.96, 50.97, 51.35, 52.06, 52.39, 52.44, 53.22, 54.19, 54.47, 56.17,
        56.21, 56.35, 57.39, 57.43, 57.52, 57.55, 57.6, 58.1, 58.19, 58.34,
        58.6, 58.88, 58.9, 58.99, 59.52, 59.75, 59.99, 60.11, 60.23, 60.43,
        60.49, 60.69, 60.92, 61.16, 61.55, 61.59, 61.68, 61.98, 62.28, 62.3,
        63.47, 63.64, 63.71, 63.84, 63.9, 63.98, 64.51, 64.61, 64.72, 64.94,
        64.96, 65.11, 65.17, 65.23, 65.26, 65.42, 65.55, 65.68, 65.91, 66.13,
        66.19, 66.25, 66.37, 66.64, 66.68, 66.78, 67.08, 67.21, 67.49, 67.51,
        67.8, 67.81, 68.06, 68.47, 68.7, 68.75, 69.08, 69.25, 69.38, 69.44,
        69.45, 69.71, 70.1, 70.29, 70.38, 70.43, 70.62, 70.68, 70.78, 71.1,
        71.16, 71.45, 71.71, 71.83, 72.16, 72.2, 72.31, 72.55, 72.66, 72.74,
        72.79, 72.85, 72.88, 73.01, 73.3, 73.46, 73.49, 73.5, 73.53, 74.28,
        74.56, 74.63, 74.7, 74.95, 75.01, 75.14, 75.28, 75.51, 75.53, 75.72,
        75.76, 75.98, 76.06, 76.14, 76.74, 76.86, 76.9, 77.37, 77.55, 77.97,
        79.1, 79.2, 80.01, 80.6, 81.51, 82.36, 83.01, 83.43
    ]
    
    # Create analyzer with your constraints
    analyzer = AnalyticalHofsteeAnalyzer(
        scores=scores_data,
        min_cutoff=48.0,
        max_cutoff=50.0,
        min_fail_rate=0.01,  # 1%
        max_fail_rate=0.07   # 7%
    )
    
    # Print detailed analysis
    analyzer.print_detailed_analysis()
    
    # Create plots
    fig, results = analyzer.plot_analytical_solution()
    plt.show()
    
    return analyzer, results

# if __name__ == "__main__":
#     analyzer, results = demo_analytical_hofstee()
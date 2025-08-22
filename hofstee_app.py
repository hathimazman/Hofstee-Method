import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar
import seaborn as sns
import io

# Set page config
st.set_page_config(
    page_title="Hofstee Cutoff Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HofsteeAnalyzer:
    """
    A class to perform Hofstee analysis for determining cutoff scores in assessments.
    """
    
    def __init__(self, scores, min_cutoff=None, max_cutoff=None, 
                 min_fail_rate=0.05, max_fail_rate=0.40):
        self.scores = np.array(scores)
        self.min_cutoff = min_cutoff if min_cutoff is not None else np.min(scores)
        self.max_cutoff = max_cutoff if max_cutoff is not None else np.max(scores)
        self.min_fail_rate = min_fail_rate
        self.max_fail_rate = max_fail_rate
        
        self.sorted_scores = np.sort(scores)
        self.n_students = len(scores)
        
    def create_empirical_cdf_points(self, n_points=100):
        """Create empirical cumulative distribution points handling duplicates properly"""
        sorted_scores = np.sort(self.scores)
        n_total = len(sorted_scores)
        
        # Get unique scores and their cumulative counts
        unique_scores, counts = np.unique(sorted_scores, return_counts=True)
        cumulative_counts = np.cumsum(counts)
        
        empirical_scores = []
        empirical_percentages = []
        
        # Add starting point (slightly below minimum)
        empirical_scores.append(np.min(sorted_scores) - 0.1)
        empirical_percentages.append(0.0)
        
        # Add points for each unique score
        for i, score in enumerate(unique_scores):
            if i == 0:
                # First score - percentage below is 0
                below_count = 0
            else:
                # Count of scores strictly below this value
                below_count = cumulative_counts[i-1]
            
            percentage_below = (below_count / n_total) * 100
            empirical_scores.append(score)
            empirical_percentages.append(percentage_below)
        
        # Add ending point (slightly above maximum)
        empirical_scores.append(np.max(sorted_scores) + 0.1)
        empirical_percentages.append(100.0)
        
        return np.array(empirical_scores), np.array(empirical_percentages)
    
    def get_smooth_cumulative_curve(self, score_range=None, method='cubic_spline'):
        """Generate smooth cumulative curve using spline interpolation"""
        if score_range is None:
            score_range = np.linspace(np.min(self.scores) - 2, np.max(self.scores) + 2, 1000)
        
        # Get empirical CDF points
        empirical_scores, empirical_percentages = self.create_empirical_cdf_points()
        
        try:
            if method == 'cubic_spline':
                # SciPy CubicSpline with natural boundary conditions
                cs = CubicSpline(empirical_scores, empirical_percentages, bc_type='natural')
                smooth_percentages = cs(score_range)
                
            elif method == 'interp1d_cubic':
                # scipy interp1d with cubic interpolation
                f = interp1d(empirical_scores, empirical_percentages, 
                           kind='cubic', bounds_error=False, 
                           fill_value=(0, 100))
                smooth_percentages = f(score_range)
                
            else:  # 'linear' fallback
                f = interp1d(empirical_scores, empirical_percentages, 
                           kind='linear', bounds_error=False, 
                           fill_value=(0, 100))
                smooth_percentages = f(score_range)
                
        except Exception as e:
            print(f"Interpolation failed ({e}), using linear fallback")
            f = interp1d(empirical_scores, empirical_percentages, 
                       kind='linear', bounds_error=False, fill_value=(0, 100))
            smooth_percentages = f(score_range)
        
        # Ensure valid cumulative properties
        smooth_percentages = np.clip(smooth_percentages, 0, 100)
        smooth_percentages = np.maximum.accumulate(smooth_percentages)
        
        return score_range, smooth_percentages, empirical_scores, empirical_percentages
    
    def calculate_hofstee_cutoff(self, method='cubic_spline'):
        """Calculate the Hofstee cutoff using smooth interpolation"""
        # Create score range within constraints
        score_range = np.linspace(self.min_cutoff, self.max_cutoff, 1000)
        
        # Get smooth cumulative curve
        _, smooth_percentages, _, _ = self.get_smooth_cumulative_curve(score_range, method)
        
        # Hofstee diagonal: failure_rate decreases linearly from max to min
        # as score increases from min_cutoff to max_cutoff
        slope = (self.max_fail_rate - self.min_fail_rate) / (self.min_cutoff - self.max_cutoff)
        diagonal_fail_rates = self.max_fail_rate + slope * (score_range - self.min_cutoff)
        diagonal_percentages = diagonal_fail_rates * 100
        
        # Find intersection point
        differences = np.abs(smooth_percentages - diagonal_percentages)
        min_idx = np.argmin(differences)
        
        cutoff_score = score_range[min_idx]
        failure_rate = smooth_percentages[min_idx] / 100
        
        return {
            'cutoff': cutoff_score,
            'failure_rate': failure_rate,
            'percentage_below': smooth_percentages[min_idx],
            'score_range': score_range,
            'smooth_percentages': smooth_percentages,
            'diagonal_percentages': diagonal_percentages,
            'intersection_difference': differences[min_idx]
        }
    
    def plot_hofstee_cumulative(self, figsize=(12, 8), method='cubic_spline', show_empirical=True):
        """Create the Hofstee plot with smooth cubic spline interpolation"""
        results = self.calculate_hofstee_cutoff(method)
        
        # Get full range smooth curve for display
        full_score_range = np.linspace(np.min(self.scores) - 2, np.max(self.scores) + 2, 1000)
        _, full_smooth_percentages, empirical_scores, empirical_percentages = \
            self.get_smooth_cumulative_curve(full_score_range, method)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the smooth cumulative curve
        ax.plot(full_score_range, full_smooth_percentages, 'b-', linewidth=3, 
               label='Cumulative % Below Score', zorder=3)
        
        # Optionally show empirical points
        if show_empirical:
            # Show every nth point to avoid overcrowding
            step = max(1, len(empirical_scores) // 20)
            ax.scatter(empirical_scores[::step], empirical_percentages[::step], 
                      color='lightblue', s=25, alpha=0.7, 
                      label='Empirical CDF Points', zorder=4)
        
        # Create Hofstee constraint rectangle
        rect_x = [self.min_cutoff, self.max_cutoff, self.max_cutoff, self.min_cutoff, self.min_cutoff]
        rect_y = [self.min_fail_rate * 100, self.min_fail_rate * 100, 
                  self.max_fail_rate * 100, self.max_fail_rate * 100, self.min_fail_rate * 100]
        
        ax.plot(rect_x, rect_y, 'r-', linewidth=2, label='Hofstee Constraints')
        ax.fill(rect_x[:-1], rect_y[:-1], alpha=0.15, color='red', label='Acceptable Region')
        
        # Draw Hofstee diagonal
        diagonal_x = [self.min_cutoff, self.max_cutoff]
        diagonal_y = [self.max_fail_rate * 100, self.min_fail_rate * 100]
        ax.plot(diagonal_x, diagonal_y, 'g--', linewidth=3, label='Hofstee Diagonal')
        
        # Mark intersection point
        intersection_score = results['cutoff']
        intersection_percentage = results['failure_rate'] * 100
        ax.plot(intersection_score, intersection_percentage, 'ro', markersize=15, 
               markeredgecolor='black', markeredgewidth=3, zorder=10)
        
        # Add constraint boundary lines
        ax.axhline(self.min_fail_rate * 100, color='orange', linestyle=':', alpha=0.8, linewidth=2,
                  label=f'Min Failure Rate ({self.min_fail_rate:.1%})')
        ax.axhline(self.max_fail_rate * 100, color='orange', linestyle=':', alpha=0.8, linewidth=2,
                  label=f'Max Failure Rate ({self.max_fail_rate:.1%})')
        ax.axvline(self.min_cutoff, color='purple', linestyle=':', alpha=0.8, linewidth=2,
                  label=f'Min Cutoff ({self.min_cutoff})')
        ax.axvline(self.max_cutoff, color='purple', linestyle=':', alpha=0.8, linewidth=2,
                  label=f'Max Cutoff ({self.max_cutoff})')
        
        # Formatting
        ax.set_xlabel('Score', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative Percentage', fontsize=13, fontweight='bold')
        ax.set_title(f'Hofstee Method: Smooth Cumulative Analysis\n({method.replace("_", " ").title()} Interpolation)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set appropriate limits
        ax.set_xlim(np.min(self.scores) - 2, np.max(self.scores) + 2)
        ax.set_ylim(-2, 102)
        
        # Add detailed annotation
        ax.annotate(f'Hofstee Cutoff: {intersection_score:.2f}\nFailure Rate: {intersection_percentage:.1f}%',
                   xy=(intersection_score, intersection_percentage), 
                   xytext=(intersection_score + 4, intersection_percentage + 15),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        return fig, results
    
    def create_summary_plots(self, figsize=(15, 10), method='cubic_spline'):
        """Create comprehensive analysis plots with improved cumulative curve"""
        results = self.calculate_hofstee_cutoff(method)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Hofstee Cutoff Analysis - Comprehensive View', fontsize=16, fontweight='bold')
        
        # Plot 1: Score distribution
        ax1.hist(self.scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(results['cutoff'], color='red', linestyle='--', linewidth=2, 
                   label=f'Hofstee Cutoff: {results["cutoff"]:.2f}')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hofstee curve with intersection (zoomed to constraint box)
        constrained_range = np.linspace(self.min_cutoff - 1, self.max_cutoff + 1, 500)
        _, constrained_smooth, _, _ = self.get_smooth_cumulative_curve(constrained_range, method)
        
        # Hofstee diagonal for constrained range
        slope = (self.max_fail_rate - self.min_fail_rate) / (self.min_cutoff - self.max_cutoff)
        diagonal_fail_rates = self.max_fail_rate + slope * (constrained_range - self.min_cutoff)
        diagonal_percentages = diagonal_fail_rates * 100
        
        ax2.plot(constrained_range, constrained_smooth, 'b-', linewidth=3, label='Smooth Cumulative Curve')
        ax2.plot(constrained_range, diagonal_percentages, 'g--', linewidth=3, label='Hofstee Diagonal')
        ax2.plot(results['cutoff'], results['failure_rate'] * 100, 'ro', markersize=12, 
                markeredgecolor='black', markeredgewidth=2,
                label=f'Intersection ({results["cutoff"]:.2f}, {results["failure_rate"]*100:.1f}%)')
        
        # Add constraint boundaries
        ax2.axhline(self.min_fail_rate * 100, color='orange', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Min Fail Rate: {self.min_fail_rate:.1%}')
        ax2.axhline(self.max_fail_rate * 100, color='orange', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Max Fail Rate: {self.max_fail_rate:.1%}')
        ax2.axvline(self.min_cutoff, color='purple', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Min Cutoff: {self.min_cutoff:.1f}')
        ax2.axvline(self.max_cutoff, color='purple', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Max Cutoff: {self.max_cutoff:.1f}')
        
        # Fill acceptable region
        rect_x = [self.min_cutoff, self.max_cutoff, self.max_cutoff, self.min_cutoff]
        rect_y = [self.min_fail_rate * 100, self.min_fail_rate * 100, 
                  self.max_fail_rate * 100, self.max_fail_rate * 100]
        ax2.fill(rect_x, rect_y, alpha=0.1, color='red', label='Acceptable Region')
        
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Cumulative Percentage')
        ax2.set_title('Hofstee Method: Constraint Region Detail')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(self.min_cutoff - 1, self.max_cutoff + 1)
        ax2.set_ylim(max(0, self.min_fail_rate * 100 - 2), min(100, self.max_fail_rate * 100 + 5))
        
        # Plot 3: Pass/Fail visualization
        pass_scores = self.scores[self.scores >= results['cutoff']]
        fail_scores = self.scores[self.scores < results['cutoff']]
        
        ax3.hist([fail_scores, pass_scores], bins=25, alpha=0.7, 
                color=['red', 'green'], label=['Fail', 'Pass'], stacked=True)
        ax3.axvline(results['cutoff'], color='black', linestyle='--', linewidth=3)
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Pass/Fail Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative curve comparison (full range)
        full_range = np.linspace(np.min(self.scores) - 2, np.max(self.scores) + 2, 1000)
        
        # Original step function
        step_percentages = []
        for score in full_range:
            below_count = np.sum(self.scores < score)
            percentage = (below_count / len(self.scores)) * 100
            step_percentages.append(percentage)
        
        # Smooth curve
        _, smooth_full, _, _ = self.get_smooth_cumulative_curve(full_range, method)
        
        ax4.plot(full_range, step_percentages, 'r-', linewidth=2, alpha=0.7, label='Step Function')
        ax4.plot(full_range, smooth_full, 'b-', linewidth=3, label=f'Smooth ({method.replace("_", " ").title()})')
        ax4.axvline(results['cutoff'], color='green', linestyle='--', linewidth=2, 
                   label=f'Cutoff: {results["cutoff"]:.2f}')
        ax4.set_xlabel('Score')
        ax4.set_ylabel('Cumulative Percentage')
        ax4.set_title('Cumulative Curve Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, results

def load_data(uploaded_file):
    """Load data from uploaded CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Hofstee Cutoff Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Standard Setting with Smooth Cubic Spline Interpolation**")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing student scores"
        )
        
        # Sample data option with your actual data
        if st.button("Use Sample Data (Testing 1920)"):
            # Your actual data
            sample_scores = [
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
            sample_df = pd.DataFrame({'scores': sample_scores})
            
            # Store in session state
            st.session_state['df'] = sample_df
            st.session_state['data_loaded'] = True
            st.success("Sample data loaded! (149 students)")
    
    # Load data
    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['df'] = df
            st.session_state['data_loaded'] = True
    elif 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        df = st.session_state['df']
    
    if df is not None:
        # Display data info
        st.subheader("📋 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        # Detect score column
        possible_score_columns = ['score', 'scores', 'Score', 'Scores', 'grade', 'Grade', 'mark', 'Mark']
        score_column = None
        
        for col in possible_score_columns:
            if col in df.columns:
                score_column = col
                break
        
        if score_column is None and len(df.columns) == 1:
            score_column = df.columns[0]
        elif score_column is None:
            score_column = st.selectbox("Select the score column:", df.columns)
        
        scores = df[score_column].dropna().values
        
        # Analyze duplicates
        unique_scores, counts = np.unique(scores, return_counts=True)
        duplicates = unique_scores[counts > 1]
        
        with col1:
            st.metric("Total Students", len(scores))
        with col2:
            st.metric("Mean Score", f"{np.mean(scores):.2f}")
        with col3:
            st.metric("Score Range", f"{np.min(scores):.1f} - {np.max(scores):.1f}")
        with col4:
            st.metric("Unique Scores", f"{len(unique_scores)} ({len(duplicates)} duplicates)")
        
        # Display duplicate analysis
        if len(duplicates) > 0:
            with st.expander("🔍 Duplicate Score Analysis"):
                st.write(f"Found {len(duplicates)} scores with duplicates:")
                duplicate_data = []
                for score in duplicates[:10]:  # Show first 10
                    count = counts[unique_scores == score][0]
                    duplicate_data.append({'Score': f"{score:.2f}", 'Count': count})
                st.dataframe(pd.DataFrame(duplicate_data), hide_index=True)
                if len(duplicates) > 10:
                    st.write(f"... and {len(duplicates) - 10} more")
        
        # Display first few rows
        with st.expander("👀 View Data Sample"):
            st.dataframe(df.head(10))
        
        # Sidebar parameters
        with st.sidebar:
            st.header("⚙️ Hofstee Parameters")
            st.markdown("Adjust these parameters based on your institutional requirements:")
            
            # Get reasonable defaults
            score_min, score_max = float(np.min(scores)), float(np.max(scores))
            score_range = score_max - score_min
            
            min_cutoff = st.number_input(
                "Minimum Cutoff Score",
                value=max(score_min, score_min + score_range * 0.1),
                min_value=float(score_min),
                max_value=float(score_max),
                step=0.5,
                help="The lowest acceptable passing score"
            )

            max_cutoff = st.number_input(
                "Maximum Cutoff Score",
                value=min(score_max, score_min + score_range * 0.3),
                min_value=float(min_cutoff),
                max_value=float(score_max),
                step=0.5,
                help="The highest acceptable passing score"
            )

            min_fail_rate = st.number_input(
                "Minimum Failure Rate (%)",
                value=1.0,
                min_value=0.0,
                max_value=50.0,
                step=0.5,
                help="Minimum acceptable percentage of students who should fail"
            ) / 100

            max_fail_rate = st.number_input(
                "Maximum Failure Rate (%)",
                value=7.0,
                min_value=float(min_fail_rate * 100),
                max_value=50.0,
                step=0.5,
                help="Maximum acceptable percentage of students who should fail"
            ) / 100
            
            # Smoothing method selection
            st.header("🎛️ Analysis Options")
            smoothing_method = st.selectbox(
                "Interpolation Method",
                ['cubic_spline', 'interp1d_cubic', 'linear'],
                index=0,
                help="Choose the smoothing method for the cumulative curve"
            )
            
            show_empirical = st.checkbox(
                "Show Empirical Points",
                value=True,
                help="Display the actual data points on the cumulative curve"
            )
            
            # Analysis button
            analyze_button = st.button("🔍 Analyze", type="primary")
        
        # Main analysis
        if analyze_button or st.session_state.get('auto_analyze', False):
            st.session_state['auto_analyze'] = True
            
            # Validate parameters
            if min_cutoff >= max_cutoff:
                st.error("❌ Minimum cutoff must be less than maximum cutoff!")
                return
            
            if min_fail_rate >= max_fail_rate:
                st.error("❌ Minimum failure rate must be less than maximum failure rate!")
                return
            
            # Perform Hofstee analysis
            analyzer = HofsteeAnalyzer(scores, min_cutoff, max_cutoff, min_fail_rate, max_fail_rate)
            
            # Results summary
            st.subheader("🎯 Analysis Results")
            results = analyzer.calculate_hofstee_cutoff(method=smoothing_method)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Hofstee Cutoff",
                    f"{results['cutoff']:.2f}",
                    help="Optimal cutoff score determined by Hofstee method"
                )
            with col2:
                st.metric(
                    "Failure Rate",
                    f"{results['failure_rate']:.1%}",
                    help="Percentage of students who would fail with this cutoff"
                )
            with col3:
                passing_students = np.sum(scores >= results['cutoff'])
                st.metric(
                    "Students Passing",
                    f"{passing_students} / {len(scores)}",
                    help="Number of students who would pass"
                )
            with col4:
                failing_students = len(scores) - passing_students
                st.metric(
                    "Students Failing",
                    f"{failing_students} / {len(scores)}",
                    help="Number of students who would fail"
                )
            
            # Quality indicators
            st.markdown("### 📈 Analysis Quality")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Intersection Precision",
                    f"{results['intersection_difference']:.4f}",
                    help="How closely the curves intersect (lower is better)"
                )
            with col2:
                smoothing_quality = "Excellent" if smoothing_method == 'cubic_spline' else "Good"
                st.metric(
                    "Smoothing Quality",
                    smoothing_quality,
                    help="Quality of the interpolation method used"
                )
            
            # Main Hofstee plot
            st.subheader("📈 Hofstee Analysis Plot")
            fig1, _ = analyzer.plot_hofstee_cumulative(method=smoothing_method, show_empirical=show_empirical)
            st.pyplot(fig1)
            
            # Comprehensive analysis
            st.subheader("📊 Detailed Analysis")
            fig2, _ = analyzer.create_summary_plots(method=smoothing_method)
            st.pyplot(fig2)
            
            # Detailed statistics
            st.subheader("📋 Detailed Statistics")
            
            pass_scores = scores[scores >= results['cutoff']]
            fail_scores = scores[scores < results['cutoff']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Overall Statistics**")
                stats_data = {
                    "Metric": ["Total Students", "Mean Score", "Median Score", "Standard Deviation", 
                              "Score Range", "Unique Scores", "Duplicate Scores"],
                    "Value": [
                        len(scores),
                        f"{np.mean(scores):.2f}",
                        f"{np.median(scores):.2f}",
                        f"{np.std(scores):.2f}",
                        f"{np.min(scores):.2f} - {np.max(scores):.2f}",
                        len(unique_scores),
                        len(duplicates)
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)
            
            with col2:
                st.markdown("**Pass/Fail Breakdown**")
                outcome_data = {
                    "Outcome": ["Passing Students", "Failing Students", "Pass Rate", 
                               "Mean (Passing)", "Mean (Failing)", "Std (Passing)", "Std (Failing)"],
                    "Value": [
                        f"{len(pass_scores)} ({len(pass_scores)/len(scores):.1%})",
                        f"{len(fail_scores)} ({len(fail_scores)/len(scores):.1%})",
                        f"{len(pass_scores)/len(scores):.1%}",
                        f"{np.mean(pass_scores):.2f}" if len(pass_scores) > 0 else "N/A",
                        f"{np.mean(fail_scores):.2f}" if len(fail_scores) > 0 else "N/A",
                        f"{np.std(pass_scores):.2f}" if len(pass_scores) > 0 else "N/A",
                        f"{np.std(fail_scores):.2f}" if len(fail_scores) > 0 else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(outcome_data), hide_index=True)
            
            # Method comparison
            st.subheader("🔬 Method Comparison")
            methods = ['cubic_spline', 'interp1d_cubic', 'linear']
            comparison_results = []
            
            for method in methods:
                method_result = analyzer.calculate_hofstee_cutoff(method=method)
                comparison_results.append({
                    'Method': method.replace('_', ' ').title(),
                    'Cutoff Score': f"{method_result['cutoff']:.2f}",
                    'Failure Rate': f"{method_result['failure_rate']:.2%}",
                    'Intersection Error': f"{method_result['intersection_difference']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_results)
            st.dataframe(comparison_df, hide_index=True)
            
            # Parameter summary
            with st.expander("🔧 Analysis Parameters Used"):
                param_data = {
                    "Parameter": ["Minimum Cutoff", "Maximum Cutoff", "Minimum Failure Rate", 
                                 "Maximum Failure Rate", "Interpolation Method", "Show Empirical Points"],
                    "Value": [
                        f"{min_cutoff:.2f}", 
                        f"{max_cutoff:.2f}", 
                        f"{min_fail_rate:.1%}", 
                        f"{max_fail_rate:.1%}",
                        smoothing_method.replace('_', ' ').title(),
                        "Yes" if show_empirical else "No"
                    ]
                }
                st.dataframe(pd.DataFrame(param_data), hide_index=True)
            
            # Export results
            st.subheader("💾 Export Results")
            
            # Create export data
            export_data = {
                'Student_ID': range(1, len(scores) + 1),
                'Score': scores,
                'Pass_Fail': ['Pass' if score >= results['cutoff'] else 'Fail' for score in scores]
            }
            export_df = pd.DataFrame(export_data)
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Pass/Fail Results (CSV)",
                    data=csv_str,
                    file_name=f"hofstee_results_{results['cutoff']:.2f}.csv",
                    mime="text/csv"
                )
            
            # Summary report
            summary_report = f"""
# Hofstee Analysis Summary Report

## Dataset Information
- **Total Students**: {len(scores)}
- **Score Range**: {np.min(scores):.2f} - {np.max(scores):.2f}
- **Mean Score**: {np.mean(scores):.2f}
- **Standard Deviation**: {np.std(scores):.2f}
- **Unique Scores**: {len(unique_scores)} ({len(duplicates)} duplicates)

## Hofstee Parameters
- **Minimum Cutoff**: {min_cutoff:.2f}
- **Maximum Cutoff**: {max_cutoff:.2f}
- **Minimum Failure Rate**: {min_fail_rate:.1%}
- **Maximum Failure Rate**: {max_fail_rate:.1%}
- **Interpolation Method**: {smoothing_method.replace('_', ' ').title()}

## Results
- **Recommended Cutoff**: {results['cutoff']:.2f}
- **Actual Failure Rate**: {results['failure_rate']:.2%}
- **Students Passing**: {len(pass_scores)} ({len(pass_scores)/len(scores):.1%})
- **Students Failing**: {len(fail_scores)} ({len(fail_scores)/len(scores):.1%})

## Quality Metrics
- **Intersection Precision**: {results['intersection_difference']:.4f}
- **Method Used**: Smooth cubic spline interpolation with proper cumulative calculation

Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            with col2:
                st.download_button(
                    label="📄 Download Summary Report",
                    data=summary_report,
                    file_name=f"hofstee_summary_{results['cutoff']:.2f}.txt",
                    mime="text/plain"
                )
    
    else:
        # Instructions when no data is loaded
        st.info("👈 Please upload a CSV or Excel file containing student scores, or use the sample data to get started.")
        
        st.markdown("""
        ## How to use this app:
        
        1. **Upload your data** - CSV or Excel file with student scores
        2. **Adjust parameters** in the sidebar:
           - Set minimum and maximum acceptable cutoff scores
           - Define acceptable failure rate range
           - Choose interpolation method (cubic spline recommended)
        3. **Click Analyze** to see results
        
        ## About the Hofstee Method:
        
        The Hofstee method is a standard-setting technique that determines optimal cutoff scores by:
        - Balancing expert judgments about acceptable score ranges
        - Considering acceptable failure rate limits
        - Finding the intersection between empirical cumulative data and expert constraints
        
        ## New Features in This Version:
        
        - **Smooth Cubic Spline Interpolation**: Creates smoother, more accurate cumulative curves
        - **Proper Cumulative Calculation**: Correctly handles duplicate scores and starts from 0%
        - **Multiple Interpolation Methods**: Compare cubic spline, cubic interpolation, and linear methods
        - **Enhanced Duplicate Handling**: Properly processes datasets with repeated scores
        - **Precision Metrics**: Shows how accurately the intersection point is calculated
        
        This method is widely used in educational assessment and certification exams.
        """)
        
        # Show sample data preview
        st.markdown("### 📊 Sample Data Preview")
        sample_scores = [43.09, 43.44, 43.57, 46.03, 49.33, 49.85, 49.88, 49.99, 50.11, 50.21]
        st.code(f"Sample scores: {sample_scores[:5]}... (149 total scores)")

if __name__ == "__main__":
    main()
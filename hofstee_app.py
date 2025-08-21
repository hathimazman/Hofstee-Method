import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d, UnivariateSpline
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
    Enhanced with smooth curve plotting capabilities.
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
        
    def calculate_failure_rate(self, cutoff):
        """Calculate the failure rate for a given cutoff score."""
        failures = np.sum(self.scores < cutoff)
        return failures / self.n_students
    
    def get_cutoff_range(self):
        """Generate a range of possible cutoff values."""
        return np.linspace(self.min_cutoff, self.max_cutoff, 1000)
    
    def get_smooth_cumulative_curve(self, score_range=None, smoothing_factor=0.3):
        """Generate smooth cumulative curve using spline interpolation."""
        if score_range is None:
            score_range = np.linspace(np.min(self.scores) - 2, np.max(self.scores) + 2, 500)
        
        # Calculate empirical cumulative percentages
        empirical_scores = []
        empirical_percentages = []
        
        # Use percentiles for better smoothing
        percentiles = np.linspace(0, 100, 50)
        for p in percentiles:
            score = np.percentile(self.scores, p)
            empirical_scores.append(score)
            empirical_percentages.append(p)
        
        # Create smooth spline
        try:
            # Use UnivariateSpline for smooth interpolation
            spline = UnivariateSpline(empirical_scores, empirical_percentages, s=smoothing_factor)
            smooth_percentages = spline(score_range)
            
            # Ensure monotonic and bounded
            smooth_percentages = np.clip(smooth_percentages, 0, 100)
            smooth_percentages = np.maximum.accumulate(smooth_percentages)
            
        except:
            # Fallback to linear interpolation if spline fails
            f = interp1d(empirical_scores, empirical_percentages, 
                        kind='cubic', bounds_error=False, fill_value='extrapolate')
            smooth_percentages = f(score_range)
            smooth_percentages = np.clip(smooth_percentages, 0, 100)
        
        return score_range, smooth_percentages
    
    def calculate_hofstee_cutoff(self):
        """Calculate the Hofstee cutoff using the compromise method."""
        cutoff_range = self.get_cutoff_range()
        failure_rates = [self.calculate_failure_rate(c) for c in cutoff_range]
        
        # Normalize cutoff scores to [0,1] for comparison with failure rates
        norm_cutoffs = (cutoff_range - self.min_cutoff) / (self.max_cutoff - self.min_cutoff)
        
        # Calculate diagonal line parameters
        slope = (self.max_fail_rate - self.min_fail_rate)
        diagonal_rates = self.max_fail_rate - slope * norm_cutoffs
        
        # Create a much finer grid for more precise intersection finding
        fine_norm_x = np.linspace(0, 1, 10000)  # Much denser sampling
        
        # Interpolate failure rates on fine grid
        fine_failure_rates = np.interp(fine_norm_x, norm_cutoffs, failure_rates)
        
        # Calculate diagonal values on fine grid
        fine_diagonal_rates = self.max_fail_rate - slope * fine_norm_x
        
        # Find intersection point
        differences = np.abs(fine_failure_rates - fine_diagonal_rates)
        min_idx = np.argmin(differences)
        
        # Get intersection coordinates
        optimal_norm_x = fine_norm_x[min_idx]
        hofstee_cutoff = self.min_cutoff + optimal_norm_x * (self.max_cutoff - self.min_cutoff)
        hofstee_fail_rate = fine_failure_rates[min_idx]
        diagonal_fail_rate = fine_diagonal_rates[min_idx]
        
        return {
            'cutoff': hofstee_cutoff,
            'failure_rate': hofstee_fail_rate,
            'diagonal_failure_rate': diagonal_fail_rate,
            'normalized_x': optimal_norm_x,
            'cutoff_range': cutoff_range,
            'failure_rates': failure_rates,
            'diagonal_rates': diagonal_rates,
            'intersection_difference': differences[min_idx]
        }
    
    def plot_hofstee_cumulative_smooth(self, figsize=(12, 8), smoothing_factor=0.3):
        """Create smooth Hofstee plot: Cumulative Percentage vs Score"""
        results = self.calculate_hofstee_cutoff()
        
        # Get smooth cumulative curve
        score_range, smooth_percentages = self.get_smooth_cumulative_curve(
            smoothing_factor=smoothing_factor
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the smooth cumulative curve with enhanced styling
        ax.plot(score_range, smooth_percentages, 'b-', linewidth=4, 
               label='Cumulative % Below Score', alpha=0.8)
        
        # Add subtle gradient effect using multiple lines
        for i, alpha in enumerate([0.1, 0.05, 0.02]):
            ax.plot(score_range, smooth_percentages, 'b-', 
                   linewidth=6-i, alpha=alpha)
        
        # Create the Hofstee bounding rectangle with rounded corners effect
        rect_corners = [
            (self.min_cutoff, self.min_fail_rate * 100),
            (self.max_cutoff, self.min_fail_rate * 100),
            (self.max_cutoff, self.max_fail_rate * 100),
            (self.min_cutoff, self.max_fail_rate * 100),
            (self.min_cutoff, self.min_fail_rate * 100)
        ]
        
        rect_x = [point[0] for point in rect_corners]
        rect_y = [point[1] for point in rect_corners]
        
        # Draw the bounding rectangle with enhanced styling
        ax.plot(rect_x, rect_y, 'r-', linewidth=3, label='Hofstee Constraints', alpha=0.8)
        ax.fill(rect_x[:-1], rect_y[:-1], alpha=0.15, color='red', label='Acceptable Region')
        
        # Draw the diagonal line with smooth styling
        diagonal_x = [self.min_cutoff, self.max_cutoff]
        diagonal_y = [self.max_fail_rate * 100, self.min_fail_rate * 100]
        ax.plot(diagonal_x, diagonal_y, 'g--', linewidth=3, label='Hofstee Diagonal', alpha=0.8)
        
        # Mark the intersection point with enhanced marker
        intersection_score = results['cutoff']
        intersection_percentage = results['failure_rate'] * 100
        
        # Multi-layer marker for better visibility
        ax.plot(intersection_score, intersection_percentage, 'o', 
               markersize=15, color='white', markeredgecolor='darkred', 
               markeredgewidth=3, alpha=0.9, zorder=10)
        ax.plot(intersection_score, intersection_percentage, 'o', 
               markersize=11, color='red', alpha=0.8, zorder=11)
        ax.plot(intersection_score, intersection_percentage, 'o', 
               markersize=6, color='white', alpha=1, zorder=12)
        
        # Add constraint boundary lines with improved styling
        constraint_style = {'linestyle': ':', 'alpha': 0.6, 'linewidth': 2}
        ax.axhline(self.min_fail_rate * 100, color='orange', 
                  label=f'Min Failure Rate ({self.min_fail_rate:.0%})', **constraint_style)
        ax.axhline(self.max_fail_rate * 100, color='orange', 
                  label=f'Max Failure Rate ({self.max_fail_rate:.0%})', **constraint_style)
        ax.axvline(self.min_cutoff, color='purple', 
                  label=f'Min Cutoff ({self.min_cutoff})', **constraint_style)
        ax.axvline(self.max_cutoff, color='purple', 
                  label=f'Max Cutoff ({self.max_cutoff})', **constraint_style)
        
        # Enhanced formatting
        ax.set_xlabel('Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Percentage', fontsize=14, fontweight='bold')
        ax.set_title('Hofstee Method: Smooth Cumulative Analysis', 
                    fontsize=16, fontweight='bold', pad=25)
        
        # Improved grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, which='minor')
        ax.minorticks_on()
        
        # Enhanced legend
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        ax.set_xlim(np.min(self.scores) - 2, np.max(self.scores) + 2)
        ax.set_ylim(-2, 102)
        
        # Enhanced annotation with better styling
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", 
                         edgecolor='orange', alpha=0.9, linewidth=2)
        ax.annotate(f'Hofstee Cutoff: {intersection_score:.2f}\nFailure Rate: {intersection_percentage:.1f}%',
                   xy=(intersection_score, intersection_percentage), 
                   xytext=(intersection_score + 4, intersection_percentage + 15),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=2, alpha=0.8),
                   bbox=bbox_props, fontsize=11, fontweight='bold')
        
        # Set background color
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        return fig, results
    
    def create_summary_plots_smooth(self, figsize=(16, 12), smoothing_factor=0.3):
        """Create comprehensive analysis plots with smooth curves"""
        results = self.calculate_hofstee_cutoff()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Hofstee Cutoff Analysis - Enhanced Smooth View', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Plot 1: Enhanced score distribution with KDE overlay
        ax1.hist(self.scores, bins=30, alpha=0.6, color='skyblue', 
                edgecolor='navy', linewidth=1.2, density=True, label='Distribution')
        
        # Add KDE overlay for smooth distribution curve
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(self.scores)
            x_kde = np.linspace(self.scores.min(), self.scores.max(), 200)
            y_kde = kde(x_kde)
            ax1.plot(x_kde, y_kde, 'navy', linewidth=3, alpha=0.8, label='Smooth Density')
        except:
            pass
        
        ax1.axvline(results['cutoff'], color='red', linestyle='--', linewidth=3, 
                   label=f'Hofstee Cutoff: {results["cutoff"]:.2f}', alpha=0.8)
        ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax1.set_title('Score Distribution with Smooth Density', fontweight='bold', fontsize=13)
        ax1.legend(frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#fafafa')
        
        # Plot 2: Smooth Hofstee curve with enhanced intersection
        cutoff_range = results['cutoff_range']
        failure_rates = np.array(results['failure_rates'])
        diagonal_rates = results['diagonal_rates']
        
        # Smooth the empirical curve
        try:
            spline_smooth = UnivariateSpline(cutoff_range, failure_rates, s=smoothing_factor*0.01)
            smooth_failure_rates = spline_smooth(cutoff_range)
        except:
            smooth_failure_rates = failure_rates
        
        # Plot smooth curves
        ax2.plot(cutoff_range, smooth_failure_rates, 'b-', linewidth=4, 
                label='Empirical Curve (Smooth)', alpha=0.8)
        ax2.plot(cutoff_range, diagonal_rates, 'g--', linewidth=3, 
                label='Hofstee Diagonal', alpha=0.8)
        
        # Enhanced intersection point
        ax2.plot(results['cutoff'], results['failure_rate'], 'o', 
                markersize=10, color='white', markeredgecolor='red', 
                markeredgewidth=3, zorder=10)
        ax2.plot(results['cutoff'], results['failure_rate'], 'o', 
                markersize=6, color='red', zorder=11)
        
        # Add constraint boundaries with better styling
        constraint_style = {'linestyle': ':', 'alpha': 0.7, 'linewidth': 2}
        ax2.axhline(self.min_fail_rate, color='orange', **constraint_style)
        ax2.axhline(self.max_fail_rate, color='orange', **constraint_style)
        ax2.axvline(self.min_cutoff, color='purple', **constraint_style)
        ax2.axvline(self.max_cutoff, color='purple', **constraint_style)
        ax2.axvline(results['cutoff'], color='red', linestyle='--', 
                   alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Failure Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Smooth Hofstee Analysis', fontweight='bold', fontsize=13)
        ax2.legend(frameon=True, fancybox=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#fafafa')
        
        # Plot 3: Enhanced Pass/Fail visualization
        pass_scores = self.scores[self.scores >= results['cutoff']]
        fail_scores = self.scores[self.scores < results['cutoff']]
        
        # Create overlapping histograms with transparency
        ax3.hist(fail_scores, bins=25, alpha=0.7, color='coral', 
                label=f'Fail ({len(fail_scores)})', edgecolor='darkred', linewidth=1)
        ax3.hist(pass_scores, bins=25, alpha=0.7, color='lightgreen', 
                label=f'Pass ({len(pass_scores)})', edgecolor='darkgreen', linewidth=1)
        
        ax3.axvline(results['cutoff'], color='black', linestyle='--', 
                   linewidth=3, alpha=0.8, label='Cutoff')
        ax3.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Pass/Fail Distribution', fontweight='bold', fontsize=13)
        ax3.legend(frameon=True, fancybox=True)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#fafafa')
        
        # Plot 4: Enhanced boxplot with swarm overlay
        box_data = [self.scores, pass_scores, fail_scores]
        labels = ['All Scores', 'Passing', 'Failing']
        colors = ['lightblue', 'lightgreen', 'coral']
        
        bp = ax4.boxplot(box_data, labels=labels, patch_artist=True, 
                        medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax4.set_title('Score Distribution by Outcome', fontweight='bold', fontsize=13)
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor('#fafafa')
        
        # Overall styling
        plt.tight_layout()
        fig.patch.set_facecolor('white')
        
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
    st.markdown('<h1 class="main-header">📊 Enhanced Hofstee Cutoff Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing student scores"
        )
        
        # Sample data option
        if st.button("Use Sample Data"):
            # Create sample data
            np.random.seed(42)
            sample_scores = np.random.normal(loc=68, scale=12, size=150)
            sample_scores = np.clip(sample_scores, 30, 100)
            sample_df = pd.DataFrame({'score': sample_scores})
            
            # Store in session state
            st.session_state['df'] = sample_df
            st.session_state['data_loaded'] = True
            st.success("Sample data loaded!")
    
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
        col1, col2, col3 = st.columns(3)
        
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
        
        with col1:
            st.metric("Total Students", len(scores))
        with col2:
            st.metric("Mean Score", f"{np.mean(scores):.2f}")
        with col3:
            st.metric("Score Range", f"{np.min(scores):.1f} - {np.max(scores):.1f}")
        
        # Display first few rows
        with st.expander("View Data Sample"):
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
                value=max(score_min, score_min + score_range * 0.3),
                step=0.5,
                help="The lowest acceptable passing score"
            )

            max_cutoff = st.number_input(
                "Maximum Cutoff Score",
                value=min(score_max, score_min + score_range * 0.7),
                step=0.5,
                help="The highest acceptable passing score"
            )

            min_fail_rate = st.number_input(
                "Minimum Failure Rate (%)",
                value=5.0,
                step=0.5,
                help="Minimum acceptable percentage of students who should fail"
            ) / 100

            max_fail_rate = st.number_input(
                "Maximum Failure Rate (%)",
                value=35.0,
                step=0.5,
                help="Maximum acceptable percentage of students who should fail"
            ) / 100
            
            # New smoothing parameter
            st.header("🎨 Plot Styling")
            smoothing_factor = st.slider(
                "Curve Smoothness",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Adjust how smooth the curves appear (higher = smoother)"
            )
            
            # Analysis button
            analyze_button = st.button("🔍 Analyze", type="primary")
        
        # Main analysis
        if analyze_button or st.session_state.get('auto_analyze', False):
            st.session_state['auto_analyze'] = True
            
            # Perform Hofstee analysis
            analyzer = HofsteeAnalyzer(scores, min_cutoff, max_cutoff, min_fail_rate, max_fail_rate)
            
            # Results summary
            st.subheader("🎯 Analysis Results")
            results = analyzer.calculate_hofstee_cutoff()
            
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
            
            # Main Hofstee plot with smooth curves
            st.subheader("📈 Enhanced Hofstee Analysis Plot")
            st.markdown("*Featuring smooth curves and professional styling*")
            fig1, _ = analyzer.plot_hofstee_cumulative_smooth(smoothing_factor=smoothing_factor)
            st.pyplot(fig1)
            
            # Comprehensive analysis with smooth styling
            st.subheader("📊 Detailed Analysis with Enhanced Visuals")
            fig2, _ = analyzer.create_summary_plots_smooth(smoothing_factor=smoothing_factor)
            st.pyplot(fig2)
            
            # Detailed statistics
            st.subheader("📋 Detailed Statistics")
            
            pass_scores = scores[scores >= results['cutoff']]
            fail_scores = scores[scores < results['cutoff']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Overall Statistics**")
                stats_data = {
                    "Metric": ["Total Students", "Mean Score", "Median Score", "Standard Deviation", "Score Range"],
                    "Value": [
                        len(scores),
                        f"{np.mean(scores):.2f}",
                        f"{np.median(scores):.2f}",
                        f"{np.std(scores):.2f}",
                        f"{np.min(scores):.2f} - {np.max(scores):.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)
            
            with col2:
                st.markdown("**Pass/Fail Breakdown**")
                outcome_data = {
                    "Outcome": ["Passing Students", "Failing Students", "Pass Rate", "Mean (Passing)", "Mean (Failing)"],
                    "Value": [
                        f"{len(pass_scores)} ({len(pass_scores)/len(scores):.1%})",
                        f"{len(fail_scores)} ({len(fail_scores)/len(scores):.1%})",
                        f"{len(pass_scores)/len(scores):.1%}",
                        f"{np.mean(pass_scores):.2f}" if len(pass_scores) > 0 else "N/A",
                        f"{np.mean(fail_scores):.2f}" if len(fail_scores) > 0 else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(outcome_data), hide_index=True)
            
            # Parameter summary
            with st.expander("🔧 Analysis Parameters Used"):
                param_data = {
                    "Parameter": ["Minimum Cutoff", "Maximum Cutoff", "Minimum Failure Rate", "Maximum Failure Rate", "Smoothing Factor"],
                    "Value": [min_cutoff, max_cutoff, f"{min_fail_rate:.1%}", f"{max_fail_rate:.1%}", f"{smoothing_factor:.1f}"]
                }
                st.dataframe(pd.DataFrame(param_data), hide_index=True)
    
    else:
        # Instructions when no data is loaded
        st.info("👈 Please upload a CSV or Excel file containing student scores, or use the sample data to get started.")
        
        st.markdown("""
        ## ✨ What's New in Enhanced Version:
        
        - **Smooth Curves**: Professional Excel-like smooth line plotting
        - **Enhanced Visuals**: Multi-layered markers, gradients, and professional styling
        - **Adjustable Smoothness**: Control curve smoothness with the slider
        - **Improved Annotations**: Better positioned labels and callouts
        - **Professional Styling**: Enhanced colors, backgrounds, and typography
        
        ## How to use this app:
        
        1. **Upload your data** - CSV or Excel file with student scores
        2. **Adjust parameters** in the sidebar:
           - Set minimum and maximum acceptable cutoff scores
           - Define acceptable failure rate range
           - Control curve smoothness for visual appeal
        3. **Click Analyze** to see results with smooth, professional plots
        
        ## About the Hofstee Method:
        
        The Hofstee method is a standard-setting technique that determines optimal cutoff scores by:
        - Balancing expert judgments about acceptable score ranges
        - Considering acceptable failure rate limits
        - Finding the intersection between empirical data and expert constraints
        
        This enhanced version provides publication-ready visualizations with smooth curves.
        """)

if __name__ == "__main__":
    main()
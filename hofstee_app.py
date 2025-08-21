import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
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
        
    def calculate_failure_rate(self, cutoff):
        """Calculate the failure rate for a given cutoff score."""
        failures = np.sum(self.scores < cutoff)
        return failures / self.n_students
    
    def get_cutoff_range(self):
        """Generate a range of possible cutoff values."""
        return np.linspace(self.min_cutoff, self.max_cutoff, 1000)
    
    def calculate_hofstee_cutoff(self):
        """Calculate the Hofstee cutoff using the compromise method."""
        cutoff_range = self.get_cutoff_range()
        failure_rates = [self.calculate_failure_rate(c) for c in cutoff_range]
        
        # Normalize cutoff scores to [0,1] for comparison with failure rates
        norm_cutoffs = (cutoff_range - self.min_cutoff) / (self.max_cutoff - self.min_cutoff)
        
        # Calculate diagonal line
        slope = (self.max_fail_rate - self.min_fail_rate)
        diagonal_rates = self.max_fail_rate - slope * norm_cutoffs
        
        # Find intersection point
        distances = np.abs(np.array(failure_rates) - diagonal_rates)
        min_idx = np.argmin(distances)
        
        hofstee_cutoff = cutoff_range[min_idx]
        hofstee_fail_rate = failure_rates[min_idx]
        
        return {
            'cutoff': hofstee_cutoff,
            'failure_rate': hofstee_fail_rate,
            'intersection_idx': min_idx,
            'cutoff_range': cutoff_range,
            'failure_rates': failure_rates,
            'diagonal_rates': diagonal_rates
        }
    
    def plot_hofstee_cumulative(self, figsize=(10, 8)):
        """Create the classic Hofstee plot: Cumulative Percentage vs Score"""
        results = self.calculate_hofstee_cutoff()
        
        # Calculate cumulative percentages
        score_range = np.linspace(np.min(self.scores) - 2, np.max(self.scores) + 2, 1000)
        cumulative_percentages = []
        
        for score in score_range:
            below_count = np.sum(self.scores < score)
            percentage = (below_count / len(self.scores)) * 100
            cumulative_percentages.append(percentage)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the cumulative curve
        ax.plot(score_range, cumulative_percentages, 'b-', linewidth=3, 
               label='Cumulative % Below Score')
        
        # Create the Hofstee bounding rectangle
        rect_corners = [
            (self.min_cutoff, self.min_fail_rate * 100),
            (self.max_cutoff, self.min_fail_rate * 100),
            (self.max_cutoff, self.max_fail_rate * 100),
            (self.min_cutoff, self.max_fail_rate * 100),
            (self.min_cutoff, self.min_fail_rate * 100)
        ]
        
        rect_x = [point[0] for point in rect_corners]
        rect_y = [point[1] for point in rect_corners]
        
        # Draw the bounding rectangle
        ax.plot(rect_x, rect_y, 'r-', linewidth=2, label='Hofstee Constraints')
        ax.fill(rect_x[:-1], rect_y[:-1], alpha=0.1, color='red', label='Acceptable Region')
        
        # Draw the diagonal line
        diagonal_x = [self.min_cutoff, self.max_cutoff]
        diagonal_y = [self.max_fail_rate * 100, self.min_fail_rate * 100]
        ax.plot(diagonal_x, diagonal_y, 'g--', linewidth=2, label='Hofstee Diagonal')
        
        # Mark the intersection point
        intersection_score = results['cutoff']
        intersection_percentage = results['failure_rate'] * 100
        ax.plot(intersection_score, intersection_percentage, 'ro', markersize=12, 
               markeredgecolor='black', markeredgewidth=2,
               label=f'Hofstee Cutoff\n({intersection_score:.2f}, {intersection_percentage:.1f}%)')
        
        # Add constraint boundary lines
        ax.axhline(self.min_fail_rate * 100, color='orange', linestyle=':', alpha=0.7,
                  label=f'Min Failure Rate ({self.min_fail_rate:.0%})')
        ax.axhline(self.max_fail_rate * 100, color='orange', linestyle=':', alpha=0.7,
                  label=f'Max Failure Rate ({self.max_fail_rate:.0%})')
        ax.axvline(self.min_cutoff, color='purple', linestyle=':', alpha=0.7,
                  label=f'Min Cutoff ({self.min_cutoff})')
        ax.axvline(self.max_cutoff, color='purple', linestyle=':', alpha=0.7,
                  label=f'Max Cutoff ({self.max_cutoff})')
        
        # Formatting
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Percentage', fontsize=12, fontweight='bold')
        ax.set_title('Hofstee Method: Cumulative Percentage vs Score', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_xlim(np.min(self.scores) - 2, np.max(self.scores) + 2)
        ax.set_ylim(-2, 102)
        
        # Add annotation
        ax.annotate(f'Hofstee Cutoff: {intersection_score:.2f}\nFailure Rate: {intersection_percentage:.1f}%',
                   xy=(intersection_score, intersection_percentage), 
                   xytext=(intersection_score + 3, intersection_percentage + 10),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig, results
    
    def create_summary_plots(self, figsize=(15, 10)):
        """Create comprehensive analysis plots"""
        results = self.calculate_hofstee_cutoff()
        
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
        
        # Plot 2: Hofstee curve with intersection
        cutoff_range = results['cutoff_range']
        failure_rates = results['failure_rates']
        diagonal_rates = results['diagonal_rates']
        
        ax2.plot(cutoff_range, failure_rates, 'b-', linewidth=2, label='Empirical Curve')
        ax2.plot(cutoff_range, diagonal_rates, 'g--', linewidth=2, label='Hofstee Diagonal')
        ax2.plot(results['cutoff'], results['failure_rate'], 'ro', markersize=10, 
                label=f'Intersection ({results["cutoff"]:.2f}, {results["failure_rate"]:.3f})')
        
        # Add constraint boundaries
        ax2.axhline(self.min_fail_rate, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Min Fail Rate: {self.min_fail_rate:.2f}')
        ax2.axhline(self.max_fail_rate, color='orange', linestyle=':', alpha=0.7, 
                   label=f'Max Fail Rate: {self.max_fail_rate:.2f}')
        ax2.axvline(self.min_cutoff, color='purple', linestyle=':', alpha=0.7, 
                   label=f'Min Cutoff: {self.min_cutoff:.2f}')
        ax2.axvline(self.max_cutoff, color='purple', linestyle=':', alpha=0.7, 
                   label=f'Max Cutoff: {self.max_cutoff:.2f}')
        
        # Add vertical line to the intersection point
        ax2.axvline(results['cutoff'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Intersection: {results["cutoff"]:.2f}')
        
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Cummulative Percentage')
        ax2.set_title('Hofstee Method: Zoom in to the boundary box')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Pass/Fail visualization
        pass_scores = self.scores[self.scores >= results['cutoff']]
        fail_scores = self.scores[self.scores < results['cutoff']]
        
        ax3.hist([fail_scores, pass_scores], bins=20, alpha=0.7, 
                color=['red', 'green'], label=['Fail', 'Pass'], stacked=True)
        ax3.axvline(results['cutoff'], color='black', linestyle='--', linewidth=2)
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Pass/Fail Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Score statistics
        ax4.boxplot([self.scores, pass_scores, fail_scores], 
                   labels=['All Scores', 'Passing', 'Failing'])
        ax4.set_ylabel('Score')
        ax4.set_title('Score Distribution by Outcome')
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
                min_value=score_min,
                max_value=score_max,
                value=max(score_min, score_min + score_range * 0.3),
                step=0.5,
                help="The lowest acceptable passing score"
            )

            max_cutoff = st.number_input(
                "Maximum Cutoff Score",
                min_value=min_cutoff,
                max_value=score_max,
                value=min(score_max, score_min + score_range * 0.7),
                step=0.5,
                help="The highest acceptable passing score"
            )

            min_fail_rate = st.number_input(
                "Minimum Failure Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="Minimum acceptable percentage of students who should fail"
            ) / 100

            max_fail_rate = st.number_input(
                "Maximum Failure Rate (%)",
                min_value=min_fail_rate * 100,
                max_value=80.0,
                value=35.0,
                step=0.5,
                help="Maximum acceptable percentage of students who should fail"
            ) / 100

            
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
            
            # Main Hofstee plot
            st.subheader("📈 Hofstee Analysis Plot")
            fig1, _ = analyzer.plot_hofstee_cumulative()
            st.pyplot(fig1)
            
            # Comprehensive analysis
            st.subheader("📊 Detailed Analysis")
            fig2, _ = analyzer.create_summary_plots()
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
                    "Parameter": ["Minimum Cutoff", "Maximum Cutoff", "Minimum Failure Rate", "Maximum Failure Rate"],
                    "Value": [min_cutoff, max_cutoff, f"{min_fail_rate:.1%}", f"{max_fail_rate:.1%}"]
                }
                st.dataframe(pd.DataFrame(param_data), hide_index=True)
    
    else:
        # Instructions when no data is loaded
        st.info("👈 Please upload a CSV or Excel file containing student scores, or use the sample data to get started.")
        
        st.markdown("""
        ## How to use this app:
        
        1. **Upload your data** - CSV or Excel file with student scores
        2. **Adjust parameters** in the sidebar:
           - Set minimum and maximum acceptable cutoff scores
           - Define acceptable failure rate range
        3. **Click Analyze** to see results
        
        ## About the Hofstee Method:
        
        The Hofstee method is a standard-setting technique that determines optimal cutoff scores by:
        - Balancing expert judgments about acceptable score ranges
        - Considering acceptable failure rate limits
        - Finding the intersection between empirical data and expert constraints
        
        This method is widely used in educational assessment and certification exams.
        """)

if __name__ == "__main__":
    main()
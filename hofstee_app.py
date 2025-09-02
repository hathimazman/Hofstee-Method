import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    Uses a smooth, monotone cubic Bézier spline for the cumulative curve (visually
    matching Excel 'Scatter with smooth lines'), and finds the intersection with the
    Hofstee diagonal on that spline.
    """
    def __init__(self, scores, min_cutoff=None, max_cutoff=None,
                 min_fail_rate=0.05, max_fail_rate=0.40):
        import numpy as np
        self.scores = np.array(scores)
        self.min_cutoff = float(min_cutoff if min_cutoff is not None else np.min(scores))
        self.max_cutoff = float(max_cutoff if max_cutoff is not None else np.max(scores))
        self.min_fail_rate = float(min_fail_rate)
        self.max_fail_rate = float(max_fail_rate)

        self.sorted_scores = np.sort(scores)
        self.n_students = len(scores)

    # ---------- ECDF (SPSS-style cumulative %) ----------
    def _ecdf(self):
        """
        Return ECDF using SPSS 'Frequencies' logic:
        - Drop missing
        - Tabulate by UNIQUE score values only
        - cumulative percentage = 100 * cumsum(freq) / N
        Returns xs (unique sorted) and ys (percent 0..100, inclusive)
        """
        import numpy as np
        arr = np.asarray(self.scores, dtype=float)
        arr = arr[~np.isnan(arr)]
        xs, counts = np.unique(arr, return_counts=True)
        cum_counts = np.cumsum(counts)
        ys = (cum_counts / cum_counts[-1]) * 100.0
        return xs.astype(float), ys.astype(float)

    # ---------- Hofstee diagonal ----------
    def _diag_y(self, x):
        """Diagonal y(x) in percent from (cmin,fmax%) to (cmax,fmin%)."""
        import numpy as np
        x = np.asarray(x, dtype=float)
        x1, x2 = self.min_cutoff, self.max_cutoff
        y1, y2 = self.max_fail_rate * 100.0, self.min_fail_rate * 100.0
        if x2 == x1:
            return np.full_like(x, (y1 + y2) / 2.0, dtype=float)
        t = (x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    # ---------- Monotone cubic slopes (Fritsch–Carlson) ----------
    def _monotone_slopes_fc(self, xs, ys, scale=1.0):
        """
        Compute node slopes m_i (dy/dx) for a monotone piecewise-cubic Hermite
        interpolation (Fritsch–Carlson). 'scale' (0..1) damps curvature if desired.
        """
        import numpy as np
        n = len(xs)
        m = np.zeros(n, dtype=float)
        if n < 2:
            return m
        dx = np.diff(xs)
        dy = np.diff(ys)
        delta = dy / dx

        # Initial interior slopes
        m[0] = delta[0]
        m[-1] = delta[-1]
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] <= 0:
                m[i] = 0.0
            else:
                m[i] = 0.5 * (delta[i - 1] + delta[i])

        # Enforce monotonicity
        for i in range(n - 1):
            if delta[i] == 0.0:
                m[i] = 0.0
                m[i + 1] = 0.0
            else:
                a = m[i] / delta[i]
                b = m[i + 1] / delta[i]
                s = a * a + b * b
                if s > 9.0:
                    tau = 3.0 / (s ** 0.5)
                    m[i] = tau * a * delta[i]
                    m[i + 1] = tau * b * delta[i]

        # Optional curvature damping to be gentler (close to Excel look)
        if scale != 1.0:
            m = scale * m
        return m

    # ---------- Convert Hermite segments to cubic Bézier control points ----------
    def _bezier_segments_from_monotone(self, xs, ys, m):
        """
        For each interval [i, i+1], produce cubic Bézier control points:
        P0=(x_i, y_i)
        P1=P0 + (Δx/3, m_i*Δx/3)
        P2=P3 - (Δx/3, m_{i+1}*Δx/3)
        P3=(x_{i+1}, y_{i+1})
        Returns list of segments, each as (P0, P1, P2, P3) with 2D tuples.
        """
        segs = []
        for i in range(len(xs) - 1):
            x0, y0 = xs[i], ys[i]
            x3, y3 = xs[i + 1], ys[i + 1]
            dx = x3 - x0
            # Tangent vectors with respect to param t in [0,1]
            P0 = (x0, y0)
            P3 = (x3, y3)
            P1 = (x0 + dx / 3.0, y0 + m[i] * dx / 3.0)
            P2 = (x3 - dx / 3.0, y3 - m[i + 1] * dx / 3.0)
            segs.append((P0, P1, P2, P3))
        return segs

    # ---------- Bézier evaluation ----------
    @staticmethod
    def _bezier_eval(P0, P1, P2, P3, t):
        """Evaluate cubic Bézier at scalar/array t in [0,1]. Returns (x(t), y(t))."""
        import numpy as np
        t = np.asarray(t, dtype=float)
        u = 1.0 - t
        b0 = u * u * u
        b1 = 3.0 * u * u * t
        b2 = 3.0 * u * t * t
        b3 = t * t * t
        x = b0 * P0[0] + b1 * P1[0] + b2 * P2[0] + b3 * P3[0]
        y = b0 * P0[1] + b1 * P1[1] + b2 * P2[1] + b3 * P3[1]
        return x, y

    # ---------- Sample the spline for plotting ----------
    def _sample_bezier(self, segments, n_per=100):
        import numpy as np
        xs_all, ys_all = [], []
        for (P0, P1, P2, P3) in segments:
            t = np.linspace(0.0, 1.0, n_per, endpoint=False)
            x, y = self._bezier_eval(P0, P1, P2, P3, t)
            xs_all.append(x)
            ys_all.append(y)
        # include last node
        xs_all.append(np.array([segments[-1][-1][0]]))
        ys_all.append(np.array([segments[-1][-1][1]]))
        return np.concatenate(xs_all), np.concatenate(ys_all)

    # ---------- Intersection of diagonal with the spline ----------
    def _intersect_diagonal_on_spline(self, segments, tol=1e-6, max_iter=60):
        """
        Find (x*, y*) on the cubic Bézier spline where y(t) == diag_y(x(t)).
        We search segment-by-segment: coarse t grid to locate a sign change, then
        refine with bisection on t. Returns (x*, y*). If none, returns closest point.
        """
        import numpy as np

        def f(P0, P1, P2, P3, t):
            x, y = self._bezier_eval(P0, P1, P2, P3, t)
            return y - self._diag_y(x)

        candidates = []
        for (P0, P1, P2, P3) in segments:
            # coarse scan
            T = np.linspace(0.0, 1.0, 256)
            vals = f(P0, P1, P2, P3, T)
            sgn = np.sign(vals)
            idx = np.where(np.diff(sgn) != 0)[0]
            if idx.size > 0:
                # refine first crossing in this segment
                i = int(idx[0])
                a, b = T[i], T[i + 1]
                fa, fb = vals[i], vals[i + 1]
                # bisection
                for _ in range(max_iter):
                    mid = 0.5 * (a + b)
                    fm = f(P0, P1, P2, P3, mid)
                    if abs(fm) <= tol:
                        a = b = mid
                        break
                    if fa * fm <= 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                t_star = 0.5 * (a + b)
                x_star, y_star = self._bezier_eval(P0, P1, P2, P3, t_star)
                return float(x_star), float(y_star)
            else:
                # no crossing; keep closest diff point as candidate
                j = int(np.argmin(np.abs(vals)))
                xj, yj = self._bezier_eval(P0, P1, P2, P3, T[j])
                candidates.append((float(xj), float(yj), float(abs(vals[j]))))

        # fallback to closest approach if no sign change anywhere
        if candidates:
            x_star, y_star, _ = min(candidates, key=lambda z: z[2])
            return x_star, y_star

        # worst-case: return middle of the full range
        xm = 0.5 * (self.min_cutoff + self.max_cutoff)
        return xm, float(self._diag_y(xm))

    # ---------- Public API ----------
    def calculate_hofstee_cutoff(self, smoothing_scale=1.0):
        """
        Calculate Hofstee cutoff using a smooth monotone cubic Bézier spline for the
        cumulative curve. 'smoothing_scale' (0..1] damps curvature if needed.
        Returns:
          cutoff (score), failure_rate (proportion), diagonal_failure_rate (prop),
          ecdf_x, ecdf_y_percent, and sampled spline for plotting (spline_x, spline_y_percent).
        """
        import numpy as np

        xs, ys = self._ecdf()  # percent
        if len(xs) < 2:
            # Degenerate fallback
            y_diag = self._diag_y(xs)
            i = int(np.argmin(np.abs(ys - y_diag)))
            return {
                'cutoff': float(xs[i]),
                'failure_rate': float(ys[i] / 100.0),
                'diagonal_failure_rate': float(y_diag[i] / 100.0),
                'ecdf_x': xs,
                'ecdf_y_percent': ys,
                'spline_x': xs,
                'spline_y_percent': ys
            }

        m = self._monotone_slopes_fc(xs, ys, scale=float(smoothing_scale))
        segments = self._bezier_segments_from_monotone(xs, ys, m)
        sx, sy = self._sample_bezier(segments, n_per=120)
        x_star, y_star = self._intersect_diagonal_on_spline(segments)

        return {
            'cutoff': float(x_star),
            'failure_rate': float(y_star / 100.0),
            'diagonal_failure_rate': float(self._diag_y(x_star) / 100.0),
            'ecdf_x': xs,
            'ecdf_y_percent': ys,
            'spline_x': sx,
            'spline_y_percent': sy
        }

    def plot_hofstee_cumulative(self, figsize=(10, 8), smoothing_scale=1.0):
        """
        Hofstee plot using the cumulative curve (percent).
        """
        import matplotlib.pyplot as plt
        results = self.calculate_hofstee_cutoff(smoothing_scale=smoothing_scale)

        xs, ys = results['spline_x'], results['spline_y_percent']  # smooth curve in percent
        fig, ax = plt.subplots(figsize=figsize)

        # Smooth cumulative curve
        ax.plot(xs, ys, linewidth=3, label='Cumulative % ≤ Score (smooth)')

        # Bounding rectangle
        rect_x = [self.min_cutoff, self.max_cutoff, self.max_cutoff, self.min_cutoff, self.min_cutoff]
        rect_y = [self.min_fail_rate*100, self.min_fail_rate*100, self.max_fail_rate*100,
                  self.max_fail_rate*100, self.min_fail_rate*100]
        ax.plot(rect_x, rect_y, 'r-', linewidth=2, label='Hofstee Constraints')
        ax.fill(rect_x[:-1], rect_y[:-1], alpha=0.1, color='red', label='Acceptable Region')

        # Diagonal line
        ax.plot([self.min_cutoff, self.max_cutoff],
                [self.max_fail_rate*100, self.min_fail_rate*100],
                'g--', linewidth=2, label='Hofstee Diagonal')

        # Intersection point (on the spline)
        x_star = results['cutoff']
        y_star = results['failure_rate'] * 100.0
        ax.plot(x_star, y_star, 'o', markersize=10, color='black',
                label=f'Cutoff: {x_star:.2f}')

        # Constraint boundaries
        ax.axhline(self.min_fail_rate*100, color='orange', linestyle=':', alpha=0.7,
                   label=f'Min Fail Rate: {self.min_fail_rate:.2f}')
        ax.axhline(self.max_fail_rate*100, color='orange', linestyle=':', alpha=0.7,
                   label=f'Max Fail Rate: {self.max_fail_rate:.2f}')
        ax.axvline(self.min_cutoff, color='purple', linestyle=':', alpha=0.7,
                   label=f'Min Cutoff: {self.min_cutoff:.2f}')
        ax.axvline(self.max_cutoff, color='purple', linestyle=':', alpha=0.7,
                   label=f'Max Cutoff: {self.max_cutoff:.2f}')

        # Labels, limits, legend
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Percentage', fontsize=12, fontweight='bold')
        ax.set_title('Hofstee: Cumulative % vs Score', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlim(min(self.scores) - 2, max(self.scores) + 2)
        ax.set_ylim(-2, 102)

        # Annotation
        ax.annotate(f'Cutoff: {x_star:.2f}\nFail Rate: {y_star:.1f}%',
                    xy=(x_star, y_star),
                    xytext=(x_star + (max(self.scores)-min(self.scores))*0.05, y_star + 10),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    fontsize=10, fontweight='bold')

        plt.tight_layout()
        return fig, results

    def create_summary_plots(self, figsize=(15, 10), smoothing_scale=1.0):
        """
        Reuses the smooth spline for the Hofstee panel, so the intersection is the
        true spline point (not a linearized proxy).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        results = self.calculate_hofstee_cutoff(smoothing_scale=smoothing_scale)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Hofstee Cutoff Analysis - Comprehensive View', fontsize=16, fontweight='bold')

        # Plot 1: Score distribution
        ax1.hist(self.scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(results['cutoff'], color='red', linestyle='--', linewidth=2,
                    label=f'Cutoff: {results["cutoff"]:.2f}')
        ax1.set_xlabel('Score'); ax1.set_ylabel('Frequency'); ax1.set_title('Score Distribution')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Plot 2: Smooth cumulative (proportion)
        sx, sy = results['spline_x'], results['spline_y_percent'] / 100.0
        ax2.plot(sx, sy, linewidth=2, label='ECDF (smooth)')
        ax2.plot([self.min_cutoff, self.max_cutoff],
                 [self.max_fail_rate, self.min_fail_rate],
                 'g--', linewidth=2, label='Hofstee Diagonal')
        ax2.plot(results['cutoff'], results['failure_rate'], 'ro', markersize=10,
                 label=f'Intersection ({results["cutoff"]:.2f}, {results["failure_rate"]:.3f})')
        ax2.axhline(self.min_fail_rate, color='orange', linestyle=':', alpha=0.7)
        ax2.axhline(self.max_fail_rate, color='orange', linestyle=':', alpha=0.7)
        ax2.axvline(self.min_cutoff, color='purple', linestyle=':', alpha=0.7)
        ax2.axvline(self.max_cutoff, color='purple', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Score'); ax2.set_ylabel('Cumulative Percentage (proportion)')
        ax2.set_title('Hofstee: Zoom into bounding box')
        ax2.set_xlim(self.min_cutoff, self.max_cutoff)
        ax2.set_ylim(self.min_fail_rate, self.max_fail_rate)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax2.grid(True, alpha=0.3)

        # Plot 3: Pass/Fail visualization
        pass_scores = self.scores[self.scores >= results['cutoff']]
        fail_scores = self.scores[self.scores < results['cutoff']]
        ax3.hist([fail_scores, pass_scores], bins=20, alpha=0.7,
                 color=['red', 'green'], label=['Fail', 'Pass'], stacked=True)
        ax3.axvline(results['cutoff'], color='black', linestyle='--', linewidth=2)
        ax3.set_xlabel('Score'); ax3.set_ylabel('Frequency'); ax3.set_title('Pass/Fail Distribution')
        ax3.legend(); ax3.grid(True, alpha=0.3)

        # Plot 4: Score statistics
        ax4.boxplot([self.scores, pass_scores, fail_scores],
                    labels=['All Scores', 'Passing', 'Failing'])
        ax4.set_ylabel('Score'); ax4.set_title('Score Distribution by Outcome'); ax4.grid(True, alpha=0.3)

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
# 📊 Hofstee Cutoff Analysis Tool

A comprehensive web application for performing Hofstee standard-setting analysis on educational assessments and certification exams. This tool helps educators and assessment professionals determine optimal pass/fail cutoff scores using the scientifically validated Hofstee method.

## 🎯 Overview

The Hofstee method is a widely-used standard-setting technique that combines expert judgment with empirical data to determine fair and defensible cutoff scores. This application provides an intuitive interface for:

- Analyzing student score distributions
- Setting expert judgment parameters
- Visualizing the Hofstee analysis process
- Determining optimal cutoff scores
- Generating comprehensive reports

## ✨ Features

### 📁 **Data Input**
- **File Upload**: Support for CSV and Excel files (.csv, .xlsx, .xls)
- **Flexible Data**: Auto-detects score columns with common naming conventions
- **Sample Data**: Built-in demo dataset for testing and learning
- **Data Validation**: Handles missing values and data quality issues

### ⚙️ **Interactive Parameters**
- **Minimum Cutoff Score**: Set the lowest acceptable passing score
- **Maximum Cutoff Score**: Set the highest acceptable passing score  
- **Failure Rate Range**: Define acceptable percentage of students who should fail
- **Real-time Adjustment**: See results update instantly as you change parameters

### 📈 **Visualizations**
- **Classic Hofstee Plot**: Cumulative percentage vs score with intersection analysis
- **Score Distribution**: Histogram showing the spread of student performance
- **Pass/Fail Analysis**: Visual breakdown of student outcomes
- **Constraint Visualization**: Clear display of expert judgment boundaries

### 📊 **Detailed Results**
- **Optimal Cutoff Score**: Scientifically determined pass/fail threshold
- **Failure Rate**: Percentage of students who would fail
- **Student Outcomes**: Detailed pass/fail statistics
- **Group Comparisons**: Mean scores for passing and failing students

## 🚀 Quick Start

### Installation

1. **Clone or download** the repository:
```bash
git clone https://github.com/yourusername/hofstee-method.git
cd hofstee-method
```

2. **Install required packages**:
```bash
pip install streamlit pandas numpy matplotlib scipy seaborn openpyxl
```

### Running the Application

```bash
streamlit run hofstee_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using Sample Data

1. Click **"Use Sample Data"** in the sidebar to load demo scores
2. Adjust the Hofstee parameters using the sliders
3. Click **"🔍 Analyze"** to see results
4. Explore the visualizations and detailed statistics

### Using Your Own Data

1. Prepare a CSV or Excel file with student scores
2. Upload the file using the file uploader
3. Select the score column if not auto-detected
4. Set your institutional parameters:
   - Minimum acceptable cutoff score
   - Maximum acceptable cutoff score
   - Acceptable failure rate range
5. Click **"🔍 Analyze"** to perform the analysis

## 📋 Data Format Requirements

Your data file should contain student scores in one of these formats:

### CSV Example:
```csv
score
75.5
82.3
68.7
91.2
...
```

### Excel Example:
| Student_ID | Score |
|------------|-------|
| 001        | 75.5  | 
| 002        | 82.3  | 
| 003        | 68.7  |

**Supported column names**: `score`, `scores`, `Score`, `Scores`

## 🧮 Understanding the Hofstee Method

### The Process

1. **Expert Judgment**: Educators define acceptable ranges for:
   - Minimum and maximum cutoff scores
   - Minimum and maximum failure rates

2. **Empirical Analysis**: The system analyzes actual student performance data

3. **Intersection Finding**: The method finds where expert constraints meet empirical data

4. **Optimal Cutoff**: The intersection point provides the most defensible cutoff score

### Key Concepts

- **Bounding Rectangle**: Defined by your min/max cutoff and failure rate parameters
- **Diagonal Line**: Represents the compromise between constraints
- **Empirical Curve**: Shows actual cumulative student performance
- **Intersection Point**: The optimal cutoff balancing all factors

## 📊 Interpreting Results

### Main Metrics
- **Hofstee Cutoff**: The recommended pass/fail score
- **Failure Rate**: Percentage of students who would fail
- **Students Passing/Failing**: Actual counts and percentages

### Plots Explanation

1. **Hofstee Plot**: Shows the intersection of expert judgment and empirical data
2. **Score Distribution**: Histogram of all student scores with cutoff line
3. **Cutoff vs Failure Rate**: Technical view of the optimization process
4. **Pass/Fail Distribution**: Visual breakdown of student outcomes

## 🔧 Customization

### Parameter Guidelines

- **Min Cutoff**: Set based on minimum competency standards
- **Max Cutoff**: Avoid setting too high to prevent excessive failures
- **Min Failure Rate**: Usually 5-10% (some students should fail)
- **Max Failure Rate**: Typically 20-40% depending on assessment difficulty

### Institutional Considerations

- **High-Stakes Exams**: Use conservative parameters
- **Formative Assessments**: Allow more flexibility
- **Certification Tests**: Align with professional standards
- **Course Grades**: Consider institutional grading policies

## 📁 File Structure

```
hofstee-analysis/
│
├── hofstee_app.py          # Main Streamlit application
├── README.md               # This documentation
└── requirements.txt        # Python dependencies
```

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **SciPy**: Scientific computing
- **Seaborn**: Statistical data visualization
- **OpenPyXL**: Excel file handling

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions, issues, or suggestions:

- **Email**: hathimazman@ukm.edu.my
- **Documentation**: Check this README for detailed guidance

## 🙏 Acknowledgments

- Dr. W. K. B. Hofstee for developing the original method
- The educational measurement community for validation research
- Contributors and beta testers who helped improve this tool

---

**Made with ❤️ for educators and assessment professionals**

*Happy analyzing! 🎉*
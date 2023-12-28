# Animal Training Dashboards
----

This project aims to streamline the process of querying, wrangling, visualizing, and tracking animal training performance metrics in the Brody Lab.

In the [code](/code) section, you'll find a variety of Python scripts and comprehensive documentation. These resources are dedicated to assessing training progress and creating visuals for presentations and lab meetings. These scripts utilize popular Python libraries for efficient data handling and visualization.

Each day, our Python notebooks automatically process new data from the lab's SQL databases. These notebooks generate two types of summary visuals for each subject: a detailed single-day plot and a broader multi-week overview. The visuals are tailored to reflect the specific goals and curricula of each training stage. An example daily plot is shown below.

![C223_2023-12-16_day_summary](https://github.com/Brody-Lab/jbreda_animal_training/assets/53059059/ad826621-72f9-46ca-9b3f-1e6f5c1d645f)

## Highlights

**Data Processing Scripts**: Includes scripts for efficiently cleaning and ingesting data from SQL databases into Pandas DataFrames, tailored for the specific needs of animal training data.

**Advanced Visualization Tools**: Features logic-based visualization techniques using Seaborn and Matplotlib for both daily and multi-week analyses, enabling detailed tracking of training progress.

**Comprehensive Documentation**: Offers detailed documentation covering script usage, data handling tips, and visual interpretation guides, facilitating easy adoption for new users.
Usage

## Usage

For guidance on querying the Bdata SQL tables, refer to the tutorial [here](https://github.com/jess-breda/DataJoint-SQL-Tutorial).

To process data from Bdata, use create_days_df.py and create_trials_df.py scripts for data ingestion and cleaning. Then, for generating visuals, refer to plot_days_df.py and plot_trials_df.py.

For details on the Delayed Match to Sample (DMS) task and its implementation in our automated training processes, visit the automated training [repository](https://github.com/jess-breda/automated_training/tree/main).


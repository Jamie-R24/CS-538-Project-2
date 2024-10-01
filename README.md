# CS-538-Project-2
Project 2 for CS 538

Task 1 – BYOD (Bring Your Own Data) [10 points]
Craft a compelling introduction for your project. Describe the dataset that you are going to use,
introduce the background, explain its significance, and share your motivation for choosing it.
You don’t have to finalize the specific problem that you are going to work on just yet but you
should have a sense of what you are targeting.
Highlight the potential impact of a successful solution and anticipate any challenges ahead. This
introduction should encapsulate the essence of your project, drawing readers in with clarity and
conviction, while setting the stage for the details to follow.

Task 2 – EDA [10 points]
Perform exploratory analysis on the data. Research online for ideas, and then show analysis on
at least five different aspects of the dataset. Analyze your findings.
Note 1: This part is open-ended, any valid analysis is fine as long as it useful for you to
understand the data better! Use visualizations when necessary.
Note 2: If you choose to combine your Project 1 datasets, make sure the EDA highlights the
integrated nature of the dataset (do not reuse the two independent EDAs from Project 1).

Task 3 – Problem Definition (Revisiting Task 1) [5 points]
Based on Task 2, revisit Task 1 to write a compelling introduction for your project that also
highlights the problem that you are going to solve and some observations to support the choice,
motivation and significance based on your EDA.

Task 4 – Preprocessing [15 points]
Perform data processing on the data. Dive deep into data preparation, ensuring that the dataset
is primed and ready for model training. Document all preprocessing steps and include them in
the report, specify what happened to the data after the step, why you performed it and if you are
going to use it moving forward.
For example:
1. Data Cleaning: Address any inconsistencies within the data. This includes dealing with
missing values, anomalies, and duplicated entries.
2. Data Transformation & Normalization: Depending on the nature and distribution of
your data, apply necessary transformations. Ensure features are on a similar scale,
making them more amenable to analysis and model training.
3. Specific Data Type Processing:
a. Text Data: Implement tokenization to break down text into smaller chunks,
remove stop words to filter out common but uninformative words, and utilize
feature extraction techniques, like TF-IDF or word embeddings, to represent text
in a form suitable for machine learning.
b. Image Data: Normalize image pixel values, ensuring they fall within a consistent
range (typically 0 to 1). Consider image augmentation techniques to artificially
enhance your dataset, creating varied representations of the same image to
improve model robustness.

Task 5 – Model Selection, Training, and Optimization [30 points]
In this task, you'll identify and train at least five distinct machine learning models apt for your
classification task. Following that, try to boost the performance of the models using suitable
techniques. For example, feature engineering, cross-validation, grid search, tuning model
hyperparameters, etc. Try at least three techniques and compare the performance with the
vanilla ones.

Task 6 – Model Evaluation [30 points]
1. Use at least four metrics to evaluate the performance of your selected models.
2. Analyze the experimental results and report your conclusion. The analysis should
compare results (using the metrics you chose above) the different pre-processing steps
(Task 4), the different models and optimization steps (Task 5) in a table/graph. Report
your observations discuss their implications (e.g., Model X with Optimization Y after
preprocessing step Z was the most/least effective because ABC…).

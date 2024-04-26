# Esguerra-CSST-104
Activities and Projects Compilation

# **Activities and Projects Compilation**

# **Introduction**

 Welcome to my Compilation of activities and project Below, you'll find a summary of my key project and activities related to computer courses, data science, machine learning and quantitative methods.


# **Table of Contents**

1.  [Computer Courses](#computer-courses)
2.  [Data Science](#data-science)
3.  [Machine Learning](#machine-learning)
4.  [Quantitative Methods](#quantitative-methods)
5.  [Activities](#activities-during-lab)
6.  [Exercises](#during-onlineclass)
7.  [Midterm](#alternative-midterm)


# **Computer Courses**

Computer courses encompass a wide range of topics, from programming and software development to network administration and cybersecurity.

# **Courses**

1. **Programming Languages:**Courses focusing on programming languages like Python, Java, C++, and others are fundamental. These courses cover syntax, data structures, algorithms, and problem-solving techniques.

2. **Web Development:** Web development courses teach HTML, CSS, JavaScript, and frameworks like React or Angular. They cover front-end and back-end development, databases, and server-side scripting.


# **Data Science**

Data Science is a multidisciplinary field that combines techniques from statistics, mathematics, computer science, and domain expertise to extract knowledge and insights from structured and unstructured data.

# **Data Science Courses**

 1. **Introduction to Data Science:** The course begins with an overview of the data science process, including data acquisition, cleaning, exploration, analysis, and interpretation.

2. **Statistical Analysis:** Students learn basic statistical concepts and techniques for analyzing data, including measures of central tendency, dispersion, hypothesis testing, and regression analysis.

3. **Data Wrangling:** This section covers techniques for data cleaning, transformation, and manipulation to prepare raw data for analysis. Topics include data cleaning, missing value imputation, data normalization, and feature engineering.

4. **Data Visualization:** Students learn how to visualize data using tools like Matplotlib, Seaborn, and Plotly. Topics include plotting histograms, scatter plots, bar charts, and heatmaps to explore and communicate insights from data.

# **Machine Learning**

Machine Learning is a subfield of artificial intelligence (AI) that focuses on developing algorithms and statistical models that enable computers to learn from and make predictions or decisions based on data.

 # **Machine Learning Courses**

 1. **Introduction to Machine Learning:** The course begins with an overview of machine learning concepts, including supervised learning, unsupervised learning, and reinforcement learning. Students learn about the basic components of a machine learning system and the types of problems that machine learning can address.

2. **Regression:** Students learn about regression analysis, a supervised learning technique used for predicting continuous outcomes. Topics include linear regression, polynomial regression, regularization techniques, and model evaluation metrics such as mean squared error and R-squared.

3. **Classification:** This section covers classification algorithms, which are used for predicting categorical outcomes. Topics include logistic regression, decision trees, support vector machines (SVM), k-nearest neighbors (KNN), and ensemble methods like random forests and gradient boosting.

4. **Clustering:** Students learn about unsupervised learning techniques for clustering similar data points together. Topics include k-means clustering, hierarchical clustering, density-based clustering, and evaluation metrics such as silhouette score and Davies–Bouldin index.

5. **Dimensionality Reduction:** This section covers techniques for reducing the dimensionality of data while preserving important information. Topics include principal component analysis (PCA), t-distributed stochastic neighbor embedding (t-SNE), and manifold learning techniques.

# **Quantitative Methods**


Quantitative Methods is a field that applies mathematical and statistical techniques to analyze and solve problems in various disciplines, including business, economics, engineering, and social sciences.


# **Quantitative Methods Courses**

1. **Descriptive Statistics:** This section covers techniques for summarizing and describing data using statistical measures such as mean, median, mode, standard deviation, variance, and percentiles. Students learn how to calculate these measures and interpret them in context.

2. **Probability Theory:** Students learn about the fundamentals of probability theory, including basic concepts such as sample spaces, events, probability distributions, and random variables. Topics include probability rules, conditional probability, independence, and Bayes' theorem.

3. **Regression Analysis:** Students learn about regression analysis, a statistical technique used for modeling the relationship between one or more independent variables and a dependent variable. Topics include simple linear regression, multiple regression, model estimation, and interpretation of regression coefficients.

4. **Time Series Analysis:** This section covers techniques for analyzing time-series data, which consists of observations collected at regular intervals over time. Topics include trend analysis, seasonal variation, autocorrelation, and forecasting methods such as moving averages and exponential smoothing.

5. **Decision Analysis:** This section covers techniques for making decisions in uncertain situations. Topics include decision trees, utility theory, risk analysis, sensitivity analysis, and Monte Carlo simulation.

# **Activities**


# **Exercises**

**Exercise 1**
@@ -0,0 +1,248 @@
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOLdIlb7uJkLE7/eGqK4cjv",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/NashEsguerra/Esguerra-CSST-104/blob/main/Exer1.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "EbpYQIBeHn_p"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**ADVANCE MACHINE LEARNING**"
      ],
      "metadata": {
        "id": "zUZ2GmERH7DG"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "1. Importing Liblaries"
      ],
      "metadata": {
        "id": "WDJBIX2BI3fj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd #data manipulation\n",
        "import matplotlib.pyplot as plt  #Visualization"
      ],
      "metadata": {
        "id": "IHdNUAdmIFRH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "2. Loading Dataset"
      ],
      "metadata": {
        "id": "aG3sh2fjJ0pJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.read_csv('Nash/sales_data_2.csv')"
      ],
      "metadata": {
        "id": "zz1qpzKMKP7g"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "3. Data Loading and Initial Analysis"
      ],
      "metadata": {
        "id": "cm6BEJYGMy15"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "statistic = data.describe()"
      ],
      "metadata": {
        "id": "KCjpK7ZCM5Zx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "4. Data Manipulation"
      ],
      "metadata": {
        "id": "a-5cFkX7OaR1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "product_revenue = data.groupby('Product')['Revenue'].sum()\n",
        "highest_revenue_product = product_revenue.idxmax()\n",
        "average_quantity_per_day = data['Quantity Sold'].mean()\n"
      ],
      "metadata": {
        "id": "t1yZunSNOhoO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Data.Visualization"
      ],
      "metadata": {
        "id": "M01Lcy83Q4Yp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "product_revenue.plot(kind='box')\n",
        "plt.xlabel('Product')\n",
        "plt.ylabel('Revenue')\n",
        "plt.title('Revenue by Product')\n",
        "plt.xticks(rotation=45)\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "3n779ZdvQ_2M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "report = \"\"\"\n",
        "Analysis Report\n",
        "1. Data Loading and Initial Analysis:\n",
        "   Basic Statistics of Dataset:\n",
        "     {}\n",
        "\n",
        "2. Data Manipulation:\n",
        "   Total Revenue for each product:\n",
        "     {}\n",
        "\n",
        "3. Data Visualization:\n",
        "   Bar char displaying revenue for each product is attached\n",
        "\n",
        "4. Insights:\n",
        "   The Product {} generates the highest revenue\n",
        "   On average, {:.2f} units are sold per day.\n",
        "\n",
        "\"\"\". format(statistic,product_revenue,highest_revenue_product,average_quantity_per_day,\n",
        "            highest_revenue_product,average_quantity_per_day)\n",
        "\n",
        "print(report)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-YFRXOy1UoJZ",
        "outputId": "e4ef4fb2-bbe1-49cc-d7e4-503919cf8249"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Analysis Report\n",
            "1. Data Loading and Initial Analysis:\n",
            "   Basic Statistics of Dataset:\n",
            "            Quantity Sold     Revenue\n",
            "count     270.000000  270.000000\n",
            "mean       53.981481  290.040444\n",
            "std        26.757823  132.995783\n",
            "min        10.000000   52.950000\n",
            "25%        31.500000  171.217500\n",
            "50%        51.000000  305.190000\n",
            "75%        77.000000  398.550000\n",
            "max       100.000000  499.000000\n",
            "\n",
            "2. Data Manipulation:\n",
            "   Total Revenue for each product:\n",
            "     Product\n",
            "Product A    25994.33\n",
            "Product B    26707.13\n",
            "Product C    25609.46\n",
            "Name: Revenue, dtype: float64\n",
            "\n",
            "3. Data Visualization:\n",
            "   Bar char displaying revenue for each product is attached\n",
            "\n",
            "4. Insights:\n",
            "   The Product Product B generates the highest revenue\n",
            "   On average, 53.98 units are sold per day.\n",
            "\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "7. Saving the Report to File"
      ],
      "metadata": {
        "id": "Ivw9ZxzTXaOM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "  with open('report.txt','w') as f:\n",
        "    f.write(report)"
      ],
      "metadata": {
        "id": "3Y3XOQi4WmhS"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}

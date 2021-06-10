"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do linear regression on each province.
Classify each country by the slope of the regression.
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
import matplotlib.pyplot as plt

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 5
MIN_CASES = 1000
NORMALIZE = True
# ------------------------------------------

# only add to this dict countries that had a difference in confirmed to deaths, see if there are any patterns
classifications = {}

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)

high_count = 0
med_count = 0
low_count = 0

high_labels = {}
med_labels = {}
low_labels = {}

c = 0

def visualize(steps, cases, labels, coefficients):
    fig = plt.figure()
    plt.title(labels[0][1])
    plt.xlabel('Day Count')
    plt.ylabel('Cases')

    p = np.poly1d(np.e**coefficients)
    plt.plot(steps, cases[0][steps[0]:], 'b.', steps, p(cases[0][steps[0]:]), 'r--')

    fig.savefig(labels[0][1])
    plt.close()

def viz(cases, labels):
    fig = plt.figure()
    plt.title('og: %s' % labels[0][1])
    plt.xlabel('Day count')
    plt.ylabel('cases')
    x = np.array([i for i in range(cases.shape[1])])
    plt.plot(x, cases[0], 'r.')
    fig.savefig('og cases: %s' % labels[0][1])
    plt.close()

#for _dist in ['minkowski', 'manhattan']:
for _dist in ['manhattan']:
    for val in np.unique(confirmed["Country/Region"]):
        # test data
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)

        # take the log of the cases
        steps = []
        case_logs = []
        for i in range(cases.shape[1]):
            if (cases[0][i] != 0):
                case_logs.append(np.log(cases[0][i]))
                steps.append(i)

        # fit regression
        coeffs = np.polyfit(np.array(steps), np.array(case_logs), 1)

        # classify by the slope of the regression line
        if (coeffs[0] >= 0.02):
            high_count += 1
            high_labels[labels[0][1]] = coeffs
        elif (coeffs[0] < 0.01):
            low_count += 1
            low_labels[labels[0][1]] = coeffs
        else:
            med_count += 1
            med_labels[labels[0][1]] = coeffs

#model Confirmed Cases in Pie Chart
print('counts (high, med, low): ', high_count, med_count, low_count)
labels = 'High', 'Medium', 'Low'
sizes = [high_count, med_count, low_count]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Classification by Rate of Increase in Cases')
fig1.savefig('Pie Chart Confirmed Classification')
plt.close()



# Calculate regression fit of deaths per country

deaths = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_deaths_global.csv')
deaths = data.load_csv_data(deaths)

high_deaths = {}
med_deaths = {}
low_deaths = {}
high_death_count = 0
med_death_count = 0
low_death_count = 0

#for _dist in ['minkowski', 'manhattan']:
for _dist in ['manhattan']:
    for val in np.unique(deaths["Country/Region"]):
        # test data
        df = data.filter_by_attribute(
            deaths, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)

        # take the log of the cases
        steps = []
        case_logs = []
        for i in range(cases.shape[1]):
            if (cases[0][i] != 0):
                case_logs.append(np.log(cases[0][i]))
                steps.append(i)

        # fit regression
        if len(steps) != 0:
            coeffs = np.polyfit(np.array(steps), np.array(case_logs), 1)
        else:
            coeffs = np.array([0, 0])

        # classify by the slope of the regression line
        if (coeffs[0] >= 0.02):
            high_death_count += 1
            high_deaths[labels[0][1]] = coeffs
        elif (coeffs[0] < 0.01):
            low_death_count += 1
            low_deaths[labels[0][1]] = coeffs
        else:
            med_death_count += 1
            med_deaths[labels[0][1]] = coeffs

#model deaths in pie chart
print('death counts (high, med, low): ', high_death_count, med_death_count, low_death_count)
labels = 'High', 'Medium', 'Low'
sizes = [high_death_count, med_death_count, low_death_count]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Classification by Rate of Increase in Deaths')
fig1.savefig('Pie Chart Deaths Classification')
plt.close()


# Compare confirmed case trajectories to death trajectories
pos_high = 0
neg_high = 0
pos_med = 0
neg_med = 0
pos_low = 0
neg_low = 0
inconclusive = 0

for country in high_labels:
    if country in high_deaths:
        pos_high += 1
    elif country in med_deaths or country in low_deaths:
        neg_high += 1
    else:
        inconclusive += 1

for country in med_labels:
    if country in med_deaths:
        pos_med += 1
    elif country in high_deaths or country in low_deaths:
        neg_med += 1
    else:
        inconclusive += 1

for country in low_labels:
    if country in low_deaths:
        pos_low += 1
    elif country in high_deaths or country in med_deaths:
        neg_low += 1
    else:
        inconclusive += 1

# model the comparisons in a Bar chart (true by type and then sum of incorrect)
labels = 'High', 'Medium', 'Low', 'Incorrect'
incorrect = neg_high + neg_med + neg_low
vals = [pos_high, pos_med, pos_low, incorrect]
fig1, ax1 = plt.subplots()
ax1.pie(vals, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Rates of Correct Trajectory Matches (Confirmed to Deaths)')
fig1.savefig('Pie Chart Trajectory Match Classification')
plt.close()


#closer look at false highs
high_low = 0
high_med = 0
high_high = 0
for country in high_labels:
    if country in low_deaths:
        if country not in classifications:
            classifications[country] = {}
        classifications[country]['Confirmed'] = 'High'
        classifications[country]['Deaths'] = 'Low'
        high_low += 1
    elif country in med_deaths:
        if country not in classifications:
            classifications[country] = {}
        classifications[country]['Confirmed'] = 'High'
        classifications[country]['Deaths'] = 'Medium'
        high_med += 1
    else:
        high_high += 1

#closer look at false mediums
med_classifications = {}
med_low = 0
med_med = 0
med_high = 0
for country in med_labels:
    if country in low_deaths:
        if country not in med_classifications:
            med_classifications[country] = {}
        med_classifications[country]['Confirmed'] = 'Medium'
        med_classifications[country]['Deaths'] = 'Low'
        med_low += 1
    elif country in med_deaths:
        med_med += 1
    else:
        med_high += 1

#closer look at false lows
low_low = 0
low_med = 0
low_high = 0
for country in low_labels:
    if country in low_deaths:
        low_low += 1
    elif country in med_deaths:
        low_med += 1
    else:
        low_high += 1

#model all classifications side by side
labels = 'High', 'Medium', 'Low'
vals_high = [high_high, high_med, high_low]
vals_med = [med_high, med_med, med_low]
labels_low = 'High', 'Low'
vals_low = [low_high, low_low]
fig2, ax2 = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
ax2[0].pie(vals_high, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax2[1].pie(vals_med, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax2[2].pie(vals_low, labels=labels_low, autopct='%1.1f%%', shadow=True, startangle=90)
fig2.suptitle('Rates of True Classifications of Trajectories (High, Med, Low)')
fig2.savefig('Pie charts correct classifications')
plt.close()


#print dict to file
with open('results/high_classifications.json', 'w') as f:
    json.dump(classifications, f, indent=4)

#print dict to file
with open('results/med_classifications.json', 'w') as f:
    json.dump(med_classifications, f, indent=4)
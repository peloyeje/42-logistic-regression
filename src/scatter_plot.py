#!/usr/bin/env python

import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

class ScatterPlot(object):
    def __init__(self, path):
        data = pd.read_csv(path)
        if "Index" in data.columns :
            data.drop("Index", axis=1, inplace=True)

        self.gryffindor = data[data['Hogwarts House'] == 'Gryffindor']
        self.slytherin = data[data['Hogwarts House'] == 'Slytherin']
        self.ravenclaw = data[data['Hogwarts House'] == 'Ravenclaw']
        self.hufflepuff = data[data['Hogwarts House'] == 'Hufflepuff']

        self.courses = data.loc[:, data.dtypes == "float64"].columns

    def ks_test(self, course_1, course_2):
        pv_gryffindor = ks_2samp(self.gryffindor.loc[:, self.courses[course_1]], self.gryffindor.loc[:, self.courses[course_1 + course_2 + 1]])[1]
        pv_slytherin = ks_2samp(self.slytherin.loc[:, self.courses[course_1]], self.slytherin.loc[:, self.courses[course_1 + course_2 + 1]])[1]
        pv_hufflepuff = ks_2samp(self.hufflepuff.loc[:, self.courses[course_1]], self.hufflepuff.loc[:, self.courses[course_1 + course_2 + 1]])[1]
        pv_ravenclaw = ks_2samp(self.ravenclaw.loc[:, self.courses[course_1]], self.ravenclaw.loc[:, self.courses[course_1 + course_2 + 1]])[1]
        pv_final = pv_gryffindor*pv_slytherin*pv_hufflepuff*pv_ravenclaw
        return(pv_final)

    def find_similar_courses(self, verbose = False):
        results = []
        for course_1 in range(len(self.courses)-1):
            for course_2 in range(len(self.courses) - course_1 - 1):
                results.append([self.courses[course_1], self.courses[course_1 + course_2 + 1],self.ks_test(course_1,course_2)])
        return np.array(results)[np.array(results)[:,2].argsort()][::-1][0]

    def plot_histogram(self):
        results = self.find_similar_courses()
        print(results[0],"and",results[1],"are the most similar courses")
        print("\n(Kolmogorov–Smirnov test applied to find similarity of distribution of the grades in two courses, in each of the four houses)")

        fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,7))
        fig.suptitle(results[0]+" and "+results[1]+" distribution", fontsize=14)
        fig.subplots_adjust(hspace=.5)

        axes[0,0].scatter(self.gryffindor[results[0]], self.gryffindor[results[0]]*0, label = results[0]);
        axes[0,0].scatter(self.gryffindor[results[1]], self.gryffindor[results[1]]*0+1, label = results[1]);

        axes[1,0].scatter(self.slytherin[results[0]], self.slytherin[results[0]]*0, label = results[0]);
        axes[1,0].scatter(self.slytherin[results[1]], self.slytherin[results[1]]*0+1, label = results[1]);

        axes[0,1].scatter(self.ravenclaw[results[0]], self.ravenclaw[results[0]]*0, label = results[0]);
        axes[0,1].scatter(self.ravenclaw[results[1]], self.ravenclaw[results[1]]*0+1, label = results[1]);

        axes[1,1].scatter(self.hufflepuff[results[0]], self.hufflepuff[results[0]]*0, label = results[0]);
        axes[1,1].scatter(self.hufflepuff[results[1]], self.hufflepuff[results[1]]*0+1, label = results[1]);

        houses = ['Gryffindor','Slytherin', 'Ravenclaw', 'Hufflepuff']
        for i in range(0,2):
            for j in range(0,2):
                plot = axes[i,j]
                idx = i+j
                plot.set_title(houses[idx]);
                plot.set_xlabel('Grades distribution')
                plot.set_ylim(-1,2)
                plot.get_yaxis().set_visible(False)
                plot.legend(loc='upper left')

        plt.show()

if __name__ == "__main__":
    path = sys.argv[1]
    ScatterPlot(path).plot_histogram()

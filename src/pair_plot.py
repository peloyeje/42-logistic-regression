#!/usr/bin/env python

import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PairPlot(object):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        if "Index" in self.data.columns :
            self.data.drop("Index", axis=1, inplace=True)

        self.courses = self.data.loc[:, self.data.dtypes == "float64"].columns
        self.n_courses = len(self.courses)

    def plot_graph(self):
        fig, axes = plt.subplots(nrows=self.n_courses, ncols=self.n_courses,figsize=(15,10))
        fig.suptitle("Pair plot matrix", fontsize=14)
        fig.subplots_adjust(hspace=.6)

        for i in range(self.n_courses):
            x_max = max(self.data[self.courses[i]])
            x_min = min(self.data[self.courses[i]])
            for j in range(self.n_courses):
                if i == j:
                    axes[i,i].hist(self.data[self.courses[i]].dropna(), color='grey')
                    axes[i,i].tick_params(axis='both', labelsize=5)
                else:
                    axes[i,j].scatter(self.data[self.courses[i]], self.data[self.courses[j]], c='blue')
                    axes[i,j].set_xlim(left=x_min, right=x_max)
                    axes[i,j].tick_params(axis='both', labelsize=5)

        for i in range(self.n_courses):
            if self.courses[i] == "Defense Against the Dark Arts":
                axes[0,i].set_title("DATDA", fontsize=7)
                axes[i,0].set_ylabel("DATDA", fontsize=7)
            elif self.courses[i] == "Care of Magical Creatures":
                axes[0,i].set_title("Care of MC", fontsize=7)
                axes[i,0].set_ylabel("Care of MC", fontsize=7)
            else:
                axes[0,i].set_title(self.courses[i], fontsize=7)
                axes[i,0].set_ylabel(self.courses[i], fontsize=7)

        plt.show()

if __name__ == "__main__":
    path = sys.argv[1]
    PairPlot(path).plot_graph()

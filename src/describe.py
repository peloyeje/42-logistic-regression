#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys

class DataAnalysis(object):
    def __init__(self, path):
        data_raw = pd.read_csv(path).drop("Index", axis = 1)
        self.data = data_raw.copy()
        self.data = self.data.loc[:, self.data.dtypes == "float64"]
        self.header = self.data.columns
        self.data = np.array(self.data.T)

    def compute_for_columns(self, function):
        row = []
        for col in self.data:
            row.append(function(col))
        return(row)

    def col_count(self, column):
        count = 0
        for i in column:
            if not np.isnan(i):
                count += 1
        return(count)

    def col_mean(self, column):
        col_sum = 0
        for i in column:
            if not np.isnan(i):
                col_sum += i
        return(col_sum/self.col_count(column))

    def col_std(self, column):
        mean = self.col_mean(column)
        return(np.sqrt(self.col_mean(np.square(column-mean))))

    def col_filter_nan(self, column):
        new_col = []
        for i in column:
            if not np.isnan(i):
                new_col.append(i)
        return(new_col)

    def col_sort(self, column):
        sorted_col = self.col_filter_nan(column)
        for i in range(1, len(sorted_col)):
            j = i-1
            nxt_element = sorted_col[i]
    # Compare the current element with next one
            while (sorted_col[j] > nxt_element) and (j >= 0):
                sorted_col[j+1] = sorted_col[j]
                j=j-1
            sorted_col[j+1] = nxt_element
        return(sorted_col)

    def col_min(self, column):
        return(self.col_sort(column)[0])

    def col_max(self, column):
        return(self.col_sort(column)[-1])

    def col_quantile(self, column, fraction):
        n = self.col_count(column)
        m = fraction*n
        sorted_col = self.col_sort(column)
        if np.floor(m) == m:
            return(sorted_col[int(m)])
        else:
            return((sorted_col[int(np.floor(m))] + sorted_col[int(np.ceil(m))])/2)

    def col_quantile_25(self, column):
        return(self.col_quantile(column,0.25))

    def col_median(self, column):
        return(self.col_quantile(column,0.5))

    def col_quantile_75(self, column):
        return(self.col_quantile(column,0.75))

    def describe_42(self):
        data_description = [np.round(self.compute_for_columns(self.col_count), 6),
                            np.round(self.compute_for_columns(self.col_mean), 6),
                           np.round(self.compute_for_columns(self.col_std), 6),
                           np.round(self.compute_for_columns(self.col_min), 6),
                           np.round(self.compute_for_columns(self.col_quantile_25), 6),
                           np.round(self.compute_for_columns(self.col_median), 6),
                           np.round(self.compute_for_columns(self.col_quantile_75), 6),
                           np.round(self.compute_for_columns(self.col_max), 6)]

        print(pd.DataFrame(data_description, index = ["Count","Mean","Std","Min","25%","50%","75%", "Max"], columns = self.header))

path = sys.argv[1]

if __name__ == "__main__":
    DataAnalysis(path).describe_42()

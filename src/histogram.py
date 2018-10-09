import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

class Histogram42(object):
    def __init__(self, path):
        data = pd.read_csv(path)
        if "Index" in data.columns :
            data.drop("Index", axis=1, inplace=True)

        self.gryffindor = data[data['Hogwarts House'] == 'Gryffindor']
        self.slytherin = data[data['Hogwarts House'] == 'Slytherin']
        self.ravenclaw = data[data['Hogwarts House'] == 'Ravenclaw']
        self.hufflepuff = data[data['Hogwarts House'] == 'Hufflepuff']

        self.courses = data.loc[:, data.dtypes == "float64"].columns

    def ks_test(self, course, verbose = True):
        list_of_tests = []
        list_of_tests.append(ks_2samp(self.gryffindor[course], self.slytherin[course])[1] < 0.05)
        list_of_tests.append(ks_2samp(self.gryffindor[course], self.ravenclaw[course])[1] < 0.05)
        list_of_tests.append(ks_2samp(self.gryffindor[course], self.hufflepuff[course])[1] < 0.05)
        list_of_tests.append(ks_2samp(self.slytherin[course], self.ravenclaw[course])[1] < 0.05)
        list_of_tests.append(ks_2samp(self.slytherin[course], self.hufflepuff[course])[1] < 0.05)
        list_of_tests.append(ks_2samp(self.ravenclaw[course], self.hufflepuff[course])[1] < 0.05)
        if list_of_tests == [True]*6:
            if verbose == True:
                print("No distribution is the same")
            return 0
        elif list_of_tests == [False]*6:
            if verbose == True:
                print("All the distributions are the same")
            return 1
        else:
            if verbose == True:
                print("Some distributions are the same")
                print(list_of_tests)
            return 0

    def find_courses(self):
        results = []
        for course in self.courses:
            if self.ks_test(course, verbose = False) == 1:
                results.append(course)
        return results

    def plot_histogram(self):
        results = self.find_courses()
        if len(results) == 0:
            print("No course has a homogeneous repartition of the grades.")
        elif len(results) == 1:
            print("The only course having a homogeneous repartition of the grades is {0}".format(results[0]))
        else:
            print("The courses that have a homogeneous repartition of the grades are {}.".format(' and '.join(results)))

        for result in results:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            fig.suptitle(result, fontsize=14)
            fig.subplots_adjust(hspace=.5)
            self.ravenclaw[result].hist(bins=20, ax=axes[0, 0], figsize=(7,5)) ; axes[0, 0].set_title('Ravenclaw'); axes[0,0].set_xlabel('Grades')
            self.gryffindor[result].hist(bins=20, ax=axes[0, 1], figsize=(7,5)); axes[0, 1].set_title('Gryffindor'); axes[0,1].set_xlabel('Grades')
            self.hufflepuff[result].hist(bins=20, ax=axes[1, 0], figsize=(7,5)); axes[1, 0].set_title('Hufflepuff'); axes[1,0].set_xlabel('Grades')
            self.slytherin[result].hist(bins=20, ax=axes[1, 1], figsize=(7,5)); axes[1, 1].set_title('Slytherin'); axes[1,1].set_xlabel('Grades')
        plt.show()

if __name__ == "__main__":
    Histogram42('../data/dataset_train.csv').plot_histogram()

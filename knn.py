# Simple KNN

import math
from collections import defaultdict


class DataPoint:
    def __init__(self, ind_vars=None, dep_var=None):
        if ind_vars is None:
            self.ind_vars = []
        else:
            self.ind_vars = ind_vars

        if dep_var is None:
            self.dep_var = 0
        else:
            self.dep_var = dep_var

    def distance(self, r):  # How close to the independent variables of one DataPoint another DataPoint's independent variables are.
        assert len(self.ind_vars) == len(r.ind_vars), f"Both DataPoints must have the same number of independent variables. {len(self.ind_vars)} != {len(r.ind_vars)}"

        summation = 0
        for x1, x2 in zip(self.ind_vars, r.ind_vars):
            summation += (x1 - x2) ** 2

        result = math.sqrt(summation)

        return result

    def __sub__(self, r):  # Redirects to self.distance().
        return self.distance(r)

    def __str__(self):
        return f"{self.ind_vars} -> {self.dep_var}"


class DPContainer:  # A container for DataPoints
    def __init__(self):
        self.data_points = []

    def add(self, i, d=None):
        if d is None:
            assert type(i) == DataPoint

            self.data_points.append(i)

        else:
            assert type(i) == list, "You must put your independent variables in a list."
            assert type(d) == int, "KNN is for categorization, not estimation."

            n = DataPoint(i, d)

            self.data_points.append(n)


    def predict(self, d, k=5):
        # if k == 1:
        #     return self.predict_ON(d)
        # elif k > 1:
        #     return self.predict_ONLogN(d, k)

        return self.predict_ONLogN(d, k)


    def predict_ON(self, d):
        d2 = DataPoint(ind_vars=d)

        result = min(self.data_points, key=lambda d1: d1 - d2)

        return result.dep_var


    def predict_ONLogN(self, d, k):
        d2 = DataPoint(ind_vars=d)

        self.data_points.sort(key=lambda d1: d1 - d2)  # Sort by distance from test value.

        vote = defaultdict(int)
        for point in self.data_points[0:k+1]:
            vote[point.dep_var] += 1

        return max(vote.keys(), key=lambda k: vote[k])


if __name__ == "__main__":
    pass
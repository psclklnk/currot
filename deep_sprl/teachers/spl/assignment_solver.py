import time
import gurobipy
import numpy as np


class AssignmentSolver:

    def __init__(self, n_new, n_target, verbose=True, max_reuse=1):
        # Build the model
        self.model = gurobipy.Model(name="assignment_solver")
        if not verbose:
            self.model.setParam("OutputFlag", 0)

        # We model the assignments as a binary variable of shape (n_new, n_cur)
        n_source = n_new + n_target
        self.assignments = self.model.addMVar((n_source, n_target), vtype=gurobipy.GRB.BINARY, name="assignments")

        # There will be (n_source + n_target) constraints, modelling that a source sample can at most be used twice and
        # that all targets must be matched by exactly one source sample
        for i in range(n_target):
            self.model.addLConstr(sum(self.assignments[j, i] for j in range(n_source)), gurobipy.GRB.EQUAL, 1,
                                  "out_%d" % i)

        for j in range(n_source):
            self.model.addLConstr(sum(self.assignments[j, i] for i in range(n_target)), gurobipy.GRB.LESS_EQUAL,
                                  max_reuse, "in_%d" % j)

        self.model.setParam("Threads", 1)
        self.n_source = n_source

    def __call__(self, old_samples, new_samples, target_samples, old_assignments=None):
        source_samples = np.concatenate((old_samples, new_samples), axis=0)
        distances = np.linalg.norm(source_samples[:, None, :] - target_samples[None, :, :], axis=-1)

        # If we do not have a full buffer, we need to extend the distances with large values to make use of the
        # fixed structure of our MLP
        if distances.shape[0] < self.n_source:
            fill_shape = (self.n_source - distances.shape[0], target_samples.shape[0])
            distances = np.concatenate((distances, 1e6 * np.ones(fill_shape)))

        self.model.setObjective(sum(distances[:, j] @ self.assignments[:, j] for j in range(target_samples.shape[0])))

        if old_assignments is not None:
            self.assignments.start = 0
            self.assignments[old_assignments[0], old_assignments[1]].start = 1

        self.model.optimize()
        return self.assignments.X


if __name__ == "__main__":
    np.random.seed(0)
    n = 200
    n_new = 50
    data1 = np.random.uniform(-1, 1, size=(n + n_new, 2))
    data2 = np.random.uniform(-1, 1, size=(n, 2))

    solver = AssignmentSolver(n_new=n_new, n_target=n, verbose=False, max_reuse=2)
    t1 = time.time()
    assignments = solver(data1[:n, :], data1[n:, :], data2, old_assignments=None)
    t2 = time.time()
    source_idxs, target_idxs = np.where(assignments)
    print("Solving took %.3e" % (t2 - t1))
    print("Cost: %.3e" % np.mean(np.linalg.norm(data1[source_idxs] - data2[target_idxs], axis=-1)))

    t1 = time.time()
    assignments = solver(data1[:n, :], data1[n:, :], data2, old_assignments=(source_idxs, target_idxs))
    t2 = time.time()
    print("Re-Solving took %.3e" % (t2 - t1))

    t1 = time.time()
    sub_n = n // 2
    off = n - sub_n
    assignments = solver(data1[:sub_n, :], data1[n:, :], data2, old_assignments=None)
    t2 = time.time()
    source_idxs, target_idxs = np.where(assignments)
    print("Reduced Solving took %.3e" % (t2 - t1))
    old_mask = source_idxs < sub_n
    points = np.concatenate((data1[source_idxs[old_mask]], data1[source_idxs[~old_mask] + off]), axis=0)
    print("Cost: %.3e" % np.mean(np.linalg.norm(points - data2[target_idxs], axis=-1)))

    selected_samples = data1[np.sum(assignments, axis=-1) > 0]

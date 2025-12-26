class TreeNode:

    def __init__(self, feature=None, child_a=None, child_b=None, child_c=None, child_d=None, split_1=None, split_2=None, split_3=None, prediction=None):
        self.feature = feature
        self.child_a = child_a
        self.child_b = child_b
        self.child_c = child_c
        self.child_d = child_d
        self.split_1 = split_1
        self.split_2 = split_2
        self.split_3 = split_3
        self.prediction = prediction
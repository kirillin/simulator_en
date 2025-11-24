import numpy as np

class Model:
    def __init__(self):
        self.q = np.zeros(1)
        self.qd = np.zeros(1)
        self.model = {}

    def some_tree(self, nb):
        model = {}

        model['NB'] = nb

        self.q = np.zeros(nb)
        self.qd = np.zeros(nb)

        model['parent'] = np.zeros(nb, dtype=int)
        model['jtype'] = []
        model['I'] = []
        model['Xtree'] = []

        for i in range(nd):
            model['jtype'].append('Rz')
            model['parent'][i] = ????

            model['I'].append(I)

        self.model = model


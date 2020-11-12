'''In this example we import the age-stratified testing data for Vo and run
each household as an independent example with exponential imports'''
from numpy import log
from tqdm import tqdm
from examples.vo.common import LikelihoodCalculation


class SerialLikelihoodCalculation(LikelihoodCalculation):
    def __init__(self):
        super().__init__()

    def _process_households(self):
        return [
            self.compute_probability(h)
            for h in tqdm(self.households)]


if __name__ == '__main__':
    calculator = SerialLikelihoodCalculation()
    # These parameters worked much better for alpha alone
    # params = linspace(0.001, 0.015, 10)
    likelihoods = [
        calculator(log(2)/tau, 1e-5) for tau in [2, 3, 7, 14, 21]]
    print(likelihoods)

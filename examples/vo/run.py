'''In this example we import the age-stratified testing data for Vo and run
each household as an independent example with exponential imports'''
from tqdm import tqdm
from examples.vo.common import LikelihoodCalculation


class SerialLikelihoodCalculation(LikelihoodCalculation):
    def __init__(self):
        super().__init__()

    def _process_households(self, model):
        return [
            model.compute_probability(h)
            for h in tqdm(self.households)]


if __name__ == '__main__':
    calculator = SerialLikelihoodCalculation()
    # These parameters worked much better for alpha alone
    # params = linspace(0.001, 0.015, 10)
    # For r
    likelihood = calculator(1e-1)

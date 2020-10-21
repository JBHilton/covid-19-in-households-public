'''In this example we import the age-stratified testing data for Vo and run
each household as an independent example with exponential imports'''
from numpy import linspace, logspace
from multiprocessing import Pool
from tqdm import tqdm
from examples.vo.common import LikelihoodCalculation


class MPLikelihoodCalculation(LikelihoodCalculation):
    def __init__(self, no_of_workers):
        super().__init__()
        self.no_of_workers = no_of_workers

    def _process_households(self, model):
        pool = Pool(self.no_of_workers)
        likelihoods = [
            value for value in tqdm(
                pool.imap(
                    model.compute_probability,
                    self.households[:40]),
                desc='Calculating',
                total=len(self.households[:40]))]
        pool.close()
        pool.join()
        return likelihoods


if __name__ == '__main__':
    calculator = MPLikelihoodCalculation(2)
    # These parameters worked much better for alpha alone
    # params = linspace(0.001, 0.015, 10)
    # For r
    params = logspace(-2, -1, 3)
    likelihoods = [calculator(p) for p in params]

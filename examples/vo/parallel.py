'''In this example we import the age-stratified testing data for Vo and run
each household as an independent example with exponential imports'''
from numpy import log
from multiprocessing import Pool
from tqdm import tqdm
from examples.vo.common import LikelihoodCalculation


class MPLikelihoodCalculation(LikelihoodCalculation):
    def __init__(self, no_of_workers):
        super().__init__()
        self.no_of_workers = no_of_workers

    def _process_households(self):
        pool = Pool(self.no_of_workers)
        likelihoods = [
            value for value in tqdm(
                pool.imap(
                    self.compute_probability,
                    self.households),
                desc='Calculating',
                total=len(self.households))]
        pool.close()
        pool.join()
        return likelihoods


if __name__ == '__main__':
    calculator = MPLikelihoodCalculation(20)
    # These parameters worked much better for alpha alone
    # params = linspace(0.001, 0.015, 10)
    # For r
    likelihoods = [
        calculator(log(2)/tau) for tau in [2, 3, 7, 14, 21]]
    print(likelihoods)

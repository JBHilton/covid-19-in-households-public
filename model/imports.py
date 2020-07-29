'''Class structure describing external importations'''
from abc import ABC
from numpy import exp
from scipy.interpolate import interp1d


class ImportModel(ABC):
    '''Abstract class for importation models'''
    def detected(self, t):
        pass

    def undetected(self, t):
        pass


class NoImportModel(ImportModel):
    def detected(self, t):
        return 0.0

    def undetected(self, t):
        return 0.0


class FixedImportModel(ImportModel):
    def __init__(
            self,
            detected_imports,
            undetected_imports):
        self.detected_imports = detected_imports
        self.undetected_imports = undetected_imports

    def detected(self, t):
        return self.detected_imports

    def undetected(self, t):
        return self.undetected_imports


class StepImportModel(ImportModel):
    def __init__(
            self,
            time,
            external_prevalance,
            detected_profile,
            undetected_profile):
        self.prevalence_interpolant = interp1d(
            time, external_prevalance,
            kind='nearest',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True)
        self.detected_profile = detected_profile
        self.undetected_profile = undetected_profile

    def detected(self, t):
        return self.prevalence_interpolant(t) * self.detected_profile

    def undetected(self, t):
        return self.prevalence_interpolant(t) * self.undetected_profile


class ExponentialImportModel(ImportModel):
    def __init__(self, r, det_profile, undet_profile):
        self.r = r
        self.det_profile = det_profile
        self.undet_profile = undet_profile

    def detected(self, t):
        return exp(self.r * t) * self.det_profile

    def undetected(self, t):
        return exp(self.r * t) * self.undet_profile

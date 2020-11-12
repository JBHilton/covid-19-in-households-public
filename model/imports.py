'''Class structure describing external importations'''
from abc import ABC
from numpy import exp, ones
from scipy.interpolate import interp1d

def import_model_from_spec(spec, det):
    text_to_type = {
        'fixed': FixedImportModel,
        'step': StepImportModel,
        'exponential': ExponentialImportModel,
        'care_home': CareHomeImportModel,
    }
    return text_to_type[spec['external_importation']['type']].make_from_spec(
        spec['external_importation'], det)


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

    @classmethod
    def make_from_spec(cls, spec, det):
        return cls()


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

    @classmethod
    def make_from_spec(cls, spec, det):
        return cls()

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

    @classmethod 
    def make_from_spec(cls, spec, det):
        r = float(spec['exponent'])
        alpha = float(spec['alpha'])
        det_profile = alpha * det
        undet_profile = alpha * (ones((10,)) - det)
        return cls(r, det_profile, undet_profile)

    def detected(self, t):
        return exp(self.r * t) * self.det_profile

    def undetected(self, t):
        return exp(self.r * t) * self.undet_profile

class CareHomeImportModel(ImportModel):
    def __init__(
            self,
            time,
            prodromal_prev,
            infected_prev):
        self.prodromal_interpolant = interp1d(
            time, prodromal_prev,
            kind='nearest',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True)
        self.infected_interpolant = interp1d(
            time, infected_prev,
            kind='nearest',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True)

    def prodromal(self, t):
        return self.prodromal_interpolant(t)

    def infected(self, t):
        return self.infected_interpolant(t)

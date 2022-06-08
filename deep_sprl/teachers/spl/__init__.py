from .self_paced_teacher_v2 import SelfPacedTeacherV2
from .self_paced_wrapper import SelfPacedWrapper
from .wasserstein_teacher import ContinuousBarycenterCurriculum

__all__ = ['SelfPacedWrapper', 'SelfPacedTeacherV2', 'ContinuousBarycenterCurriculum']

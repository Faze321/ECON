from .q_learner import ECONLearner
from .noop_learner import NoOpLearner

REGISTRY = {
    "q_learner": ECONLearner,
    "noop_learner": NoOpLearner,
} 

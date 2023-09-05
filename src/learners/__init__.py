from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
from .maven_learner import MavenLearner
from .sia_learner import SIALearner
from .entropy_learner import EntropyLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["maven_learner"] = MavenLearner
REGISTRY["sia_learner"] = SIALearner
REGISTRY["max_entropy_learner"] = EntropyLearner


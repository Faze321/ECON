from .basic_mac import LLMBasicMAC, BasicMAC
from .baseline_mac import BaselineMAC

REGISTRY = {
    "basic_mac": LLMBasicMAC,
    "baseline_mac": BaselineMAC,
}

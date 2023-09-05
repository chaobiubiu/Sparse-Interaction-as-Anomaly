REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_policy import BasicMAC as PolicyMAC
from .central_basic_controller import CentralBasicMAC
from .vf_controller import BasicVFMAC
from .maven_controller import MavenMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["policy"] = PolicyMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["vf_mac"] = BasicVFMAC
REGISTRY["maven_mac"] = MavenMAC

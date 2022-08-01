from models.E2_SGRL import E2_SGRL
from models.RLGRL import RLGL
from models.SGRL import SGRL, GCN
from models.RLG import RLG


method_dict = {
'E2_SGRL':E2_SGRL,
'GCN':GCN,
'RLGL':RLGL,
'SGRL':SGRL,
'RLG':RLG,
}

def getmodel(name):
    return method_dict[name.upper()]
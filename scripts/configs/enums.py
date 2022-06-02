import enum

class Dataset_name(str, enum.Enum):
    Cloth = 'cloth-mask'
    NoMask = 'no-mask'
    Surgical = 'surgical-mask'
    N95 = 'n95-mask'
    Improper = 'improper-mask'
    

import os
from typing import ClassVar, List
from dataclasses import dataclass


def get_torchair_decode_gear_list():
    list_str = os.getenv('PTA_TORCHAIR_DECODE_GEAR_LIST', None)
    if list_str is None:
        return [4]
    return [int(x) for x in list_str.split(',')]


@dataclass
class EnvVar:
    decode_gear_list: ClassVar[List[int]] = get_torchair_decode_gear_list()


ENV = EnvVar()

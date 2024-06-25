import os
from typing import ClassVar, List
from dataclasses import dataclass


def get_torchair_decode_gear_list():
    list_str = os.getenv('PTA_TORCHAIR_DECODE_GEAR_LIST', None)
    if list_str is None:
        return [4]
    gear_list = []
    for x in list_str.split(','):
        gear = int(x)
        if gear == 1:
            raise ValueError("PTA_TORCHAIR_DECODE_GEAR_LIST not support 1")
        gear_list.append(gear)
    return gear_list


@dataclass
class EnvVar:
    decode_gear_list: ClassVar[List[int]] = get_torchair_decode_gear_list()


ENV = EnvVar()

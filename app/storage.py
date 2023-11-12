from typing import Tuple

import pandas as pd
import numpy as np


class Storage:
    data: pd.DataFrame
    _max_coord: Tuple[int, int]

    def __init__(self, size: int, max_coord: Tuple[int, int]):
        self._max_coord = max_coord
        self.data = pd.DataFrame({
            'x': np.random.randint(low=0, high=max_coord[0], size=size),
            'y': np.random.randint(low=0, high=max_coord[1], size=size),
            'vx': np.random.randint(low=-10, high=10, size=size),
            'vy': np.random.randint(low=-10, high=10, size=size),
            'radius': np.repeat(5, size),
        })

    def move(self) -> None:
        self.data['x'] += self.data['vx']
        self.data['y'] += self.data['vy']

        self.data.loc[self.data['x'] < 0, 'vx'] += 10
        self.data.loc[self.data['y'] < 0, 'vy'] += 10

        self.data.loc[self.data['x'] > self._max_coord[0], 'vx'] -= 10
        self.data.loc[self.data['y'] > self._max_coord[1], 'vy'] -= 10

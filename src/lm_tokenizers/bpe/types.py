from typing import TypeAlias
from ..base import Token

TokenPair: TypeAlias = tuple[Token, Token]
WeightedChunk: TypeAlias = tuple[list[Token], int]





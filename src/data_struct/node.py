from typing import Union
class Var:
    def __init__(self, name: str) -> None:
        '''
        name: str
        '''
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Var):
            return self.name == other.name
        return self.__str__() == other
    
    def __hash__(self) -> int:
        return hash(self.name)
    
class Node:
    def __init__(self, name: Union[str, Var], t: int) -> None:
        '''
        name: str | Var
        t: int
        '''
        self.name = Var(name) if isinstance(name, str) else name
        self.t = int(t)
        self.gstr = f"{self.name}_{self.t}"
    
    def __str__(self) -> str:
        return f"{self.name}_{self.t}"

    def __repr__(self) -> str:
        return f"{self.name}_{self.t}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Node):
            return (self.name == other.name) and (self.t == other.t)
        return self.__str__() == other
    
    def __hash__(self) -> int:
        return hash((self.name, self.t))
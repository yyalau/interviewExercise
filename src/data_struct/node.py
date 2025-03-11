from typing import Union
class Var:
    def __init__(self, name: str) -> None:
        '''
        name: str
        '''
        assert isinstance(name, str), f"Expected name to be of type str, got {type(name)}"
        assert len(name) !=0 , "Name cannot be empty."
        assert "_" not in name, "Name cannot contain '_'."
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
        assert isinstance(name, (str, Var)), f"Expected name to be of type str or Var, got {type(name)}"

        if isinstance(t, str):
            t = int(t)
        assert isinstance(t, int), f"Expected t to be of type int, got {type(t)}"
        assert t >= 0, f"Expected t to be greater than or equal to 0, got {t}"
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
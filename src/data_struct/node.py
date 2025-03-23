from typing import Union

class Var:
    def __init__(self, name: str) -> None:
        '''
        Represents a variable in the graph, such as A, B, C, etc., without a time index.
        
        Parameters:
        -----------
        name: str
            The name of the variable.
        '''
        # Ensure the name is a string
        assert isinstance(name, str), f"Expected name to be of type str, got {type(name)}"
        # Ensure the name is not empty
        assert len(name) != 0, "Name cannot be empty."
        # Ensure the name does not contain underscores
        assert "_" not in name, "Name cannot contain '_'."
        self.name = name

    def __str__(self) -> str:
        # String representation of the variable
        return self.name

    def __repr__(self) -> str:
        # Representation of the variable for debugging
        return self.name

    def __eq__(self, other: object) -> bool:
        # Equality check: compares the name of the variable
        if isinstance(other, Var):
            return self.name == other.name
        return self.__str__() == other
    
    def __hash__(self) -> int:
        # Hash function to allow usage in sets and as dictionary keys
        return hash(self.name)
    
class Node:
    def __init__(self, name: Union[str, Var], t: int) -> None:
        '''
        Represents a variable at a specific time index in the graph, such as A_0, B_1, C_2, etc.
        
        Parameters:
        -----------
        name: str | Var
            The variable's name.
        t: int
            The time index.
        '''
        # Ensure the name is either a string or an instance of Var
        assert isinstance(name, (str, Var)), f"Expected name to be of type str or Var, got {type(name)}"

        # Convert time index to integer if it's a string
        if isinstance(t, str):
            t = int(t)
        # Ensure the time index is an integer and non-negative
        assert isinstance(t, int), f"Expected t to be of type int, got {type(t)}"
        assert t >= 0, f"Expected t to be greater than or equal to 0, got {t}"
        
        # If name is a string, convert it to a Var instance
        self.name = Var(name) if isinstance(name, str) else name
        self.t = int(t)
        # Generate a string representation of the node (e.g., A_0)
        self.gstr = f"{self.name}_{self.t}"
    
    def __str__(self) -> str:
        # String representation of the node
        return f"{self.name}_{self.t}"

    def __repr__(self) -> str:
        # Representation of the node for debugging
        return f"{self.name}_{self.t}"

    def __eq__(self, other: object) -> bool:
        # Equality check: compares both the variable name and the time index
        if isinstance(other, Node):
            return (self.name == other.name) and (self.t == other.t)
        return self.__str__() == other
    
    def __hash__(self) -> int:
        # Hash function to allow usage in sets and as dictionary keys
        return hash((self.name, self.t))
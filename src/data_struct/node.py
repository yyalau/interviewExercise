class Var:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Var):
            return self.name == other.name
        return self.__str__() == other
    
    def __hash__(self):
        return hash(self.name)
    
class Node:
    def __init__(self, name, t):
        self.name = Var(name) if isinstance(name, str) else name
        self.t = int(t)
        self.gstr = f"{self.name}_{self.t}"
    def __str__(self):
        return f"{self.name}_{self.t}"

    def __repr__(self):
        return f"{self.name}_{self.t}"

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.name == other.name) and (self.t == other.t)
        return self.__str__() == other
    
    def __hash__(self):
        return hash((self.name, self.t))
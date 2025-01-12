from collections import defaultdict

class IntervLog:
    def __init__(self):
        # along time
        self.data = defaultdict(list)
        self.keys = {
            "impv": 0,
            "i_set": 1,
            "i_level": 2,
            "y_values": 3,
            "idx": 4,
        }
        self.nT = 0
        
        self.opt_data = defaultdict(list)   
        
    def get_opt(self, key = "impv"):
        
        get_max = lambda x: max(x, key = lambda x: x[self.keys[key]])
        
        opt = self.opt_data[key]
        
        if not self.data:
            return None
        
        # import ipdb; ipdb.set_trace()
        
        if not opt:        
            for t in range(self.nT):
                opt.append(get_max(self.data[t]))            
            return opt
        
        
        for t in range(len(opt) -1, self.nT):          
            opt.append(get_max(self.data[t]))
        return opt

    def update_y(self, t, y_values):
        
        for k, v in self.opt_data.items():
            v[t][self.keys["y_values"]] = y_values
                

    def update(self, t, *, impv,  y_values, i_set, i_level):
        
        assert t +1 >= self.nT, "Time should be greater than or equal to the current time"

        self.nT = max(self.nT, t + 1)
        
        self.data[t].append([impv, i_set, i_level, y_values])
        
        
if __name__ == "__main__":
    il = IntervLog()
    il.update(0, impv = 0.5, y_values = None, i_set = "X", i_level = 0.1)
    il.update(0, impv = 0.6, y_values = None, i_set = "X", i_level = 0.2)
    il.update(0, impv = 0.7, y_values = None, i_set = "X", i_level = 0.1)
    il.update(1, impv = 0.8, y_values = None, i_set = "X", i_level = 0.4)
    il.update(1, impv = 0.9, y_values = None, i_set = "X", i_level = 0.5)

    print(il.get_opt("impv"))
    
    il.update(1, impv = 0.9, y_values = None, i_set = "X", i_level = 0.5)
    # does not update afterall
    # il.update(0, impv = 0.9, y_values = None, i_set = "X", i_level = 0.5)
    print(il.get_opt("impv"))
    print(il.get_opt("i_level"))
    il.update_y(1,  0.9)
    print(il.get_opt("impv"))
    print(il.data)
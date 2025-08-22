from copy import deepcopy 
class HistoryList:
    @property
    def records(self):
        return self._records
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,value):
        self._name=value

    def __init__(self, name=None, liveList=None):
        self._records=[]
        self._name=name
        if (liveList is not None):
            self.liveList=liveList
    
    def register(self, liveList):
        self.liveList=liveList
    
    def record(self):
        if (len(self.liveList)):
            self._records.append(deepcopy(self.liveList))



from copy import deepcopy 
class HistoryList:
    @property
    def records(self):
        return self._records

    def __init__(self):
        self._records=[]
    
    def register(self, liveList):
        self.liveList=liveList
    
    def record(self):
        if (len(self.liveList)):
            self._records.append(deepcopy(self.liveList))



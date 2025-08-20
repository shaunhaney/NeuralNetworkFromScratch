# This class is a list that allows other classes to register to be notified 
# when the list changes.  When the list changes, it notifies its subscribers by 
# calling notify() on them with a current copy of itself. 

class ObservableList(list):

    def __init__(self):
        super().__init__()

    def __init__(self, iterable):
        super().__init__(iterable)
    
    @property
    def listeners(self):
        return self._listeners

    @listeners.setter
    def listeners(self, value):
        self._listeners = value

    @listeners.deleter
    def listeners(self):
        del self._listeners

    def addListener(self, subscriber):
        if self.listeners is None:
            self.listeners=[]
        self.listeners.append(subscriber)
    
    def notify(self,snapshot):
        if (self.contains(snapshot)):
            for listener in self.listeners:
                listener.notify(self.copy())

    def addListenerForItem(self,item):
        if (isinstance(item,type(self))):
            item.addListener(self)

    def notifyListeners(self):
        for listener in self.listeners:
            listener.notify(self.copy()) 

    def __setitem__(self, index, item):
        self.addListenerForItem(item)
        super().__setitem__(index,item)
        self.notifyListeners()

    def insert(self, index, item):
        self.addListenerForItem(item)
        super().insert(index, item)
        self.notifyListeners()

    def append(self, item):
        self.addListenerForItem(item)
        super().append(item)
        self.notifyListeners()


class historyOf1:
    def record(self): 
        o=ObservableList()
        o.append(0)
        o.append(1)
        o.append(2)
        0[1]=3

    def notify(self,l):
        print(l)

h1=historyOf1()
h1.record() 
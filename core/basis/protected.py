class ProtectedClass(type):
    def __setattr__(self, key: str, value):
        if (not key.startswith('_')) and (getattr(self, key, Ellipsis) is not Ellipsis):
            raise Exception(f'{self} is protected: public attribute {key} is already set to {value}')
        super(ProtectedClass, self).__setattr__(key, value)


class ProtectedObject(object):
    def __setattr__(self, key: str, value):
        if (not key.startswith('_')) and (getattr(self, key, Ellipsis) is not Ellipsis):
            raise Exception(f'{self} is protected: public attribute {key} is already set to {value}')
        super(ProtectedObject, self).__setattr__(key, value)


class Protected(ProtectedObject, metaclass=ProtectedClass):
    pass

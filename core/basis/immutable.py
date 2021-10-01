class ImmutableClass(type):
    def __setattr__(self, key, value):
        tmp = getattr(self, key, Ellipsis)
        if tmp is not Ellipsis:
            raise Exception(f'{self} is immutable: {key} is already set to {value}')
        super(ImmutableClass, self).__setattr__(key, value)


class ImmutableObject(object):
    def __setattr__(self, key, value):
        tmp = getattr(self, key, Ellipsis)
        if tmp is not Ellipsis:
            raise Exception(f'{self} is immutable: {key} is already set to {tmp}')
        super(ImmutableObject, self).__setattr__(key, value)


class Immutable(ImmutableObject, metaclass=ImmutableClass):
    pass


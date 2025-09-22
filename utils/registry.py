class Registry(object):
    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
                name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, name=None):
        """
        Register the given object under the specified name or obj.__name__.
        
        Args:
            name (str, optional): The name to register the object under.
                                If None, uses the object's __name__ attribute.
        
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        def deco(func_or_class):
            reg_name = name if name is not None else func_or_class.__name__
            self._do_register(reg_name, func_or_class)
            return func_or_class
        
        return deco

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret

    def get_all_reg(self):
        return self._obj_map.keys()
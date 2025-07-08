class ClassRegistry:
    def __init__(self):
        self.classes = dict()
        self.args = dict()
        self.arg_keys = None

    def __getitem__(self, item):
        return self.classes[item]

    def add_to_registry(self, name):
        def add_class_by_name(cls):
            self.classes[name] = cls
            return cls

        return add_class_by_name

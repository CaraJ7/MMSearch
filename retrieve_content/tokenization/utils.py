import json
import pickle


class PickleWriteable:
    """Mixin for persisting an instance with pickle."""

    def save(self, path):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        except (pickle.PickleError, OSError) as e:
            raise IOError('Unable to save {} to path: {}'.format(self.__class__.__name__, path)) from e

    @classmethod
    def load(cls, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, OSError) as e:
            raise IOError('Unable to load {} from path: {}'.format(cls.__name__, path)) from e


class JsonWriteable:
    """Mixin for persisting an instance with json."""

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            keyword_args = json.load(f)
        return cls(**keyword_args)

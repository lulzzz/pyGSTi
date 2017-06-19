from functools import partial, wraps

# note that this decorator ignores **kwargs
def memoize(obj):
    cache = obj.cache = {}

    #@wraps(obj)
    def memoizer(*args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError('Cannot currently memoize on kwargs')
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer

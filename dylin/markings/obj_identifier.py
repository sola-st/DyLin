from typing import Optional
import weakref
from collections import defaultdict
from uuid import UUID, uuid4

class HeapMirror():
    def __init__(self):
        self.activate_weak = True
        self.mirrored_objects = {}
        self.hashable_fallback = defaultdict(uuid4)
        self.id_fallback = defaultdict(uuid4)
        self.clean_callbacks = []

    def add_clean_callback(self, func):
        self.clean_callbacks.append(func)

    def _clean(self, internal_key):
        for callback in self.clean_callbacks:
            callback(self.mirrored_objects[internal_key]["uuid"].int)
        del self.mirrored_objects[internal_key]

    def contains(self, obj: any):
        if id(obj) in self.mirrored_objects:
            return self.mirrored_objects[id(obj)]["uuid"]
        try:
            h = object.__hash__(obj)
            res = self.hashable_fallback.get(h)
            if res is None:
                raise Exception()
            return res
        except Exception:
            if id(obj) in self.id_fallback:
                return self.id_fallback[id(obj)]
        return None

    def get_ref(self, uuid) -> Optional[weakref.ReferenceType]:
        """
        Use with caution, pretty slow with current implementation.
        Only supports mirrored objects for now, no fallbacks i.e. self.id_fallback and self.hashable_fallback
        """

        for _id in self.mirrored_objects:
            mirrored = self.mirrored_objects[_id]
            if mirrored["uuid"].int == uuid:
                return mirrored["ref"]
        return None

    def getId(self, obj: any) -> Optional[str]:
        if self.activate_weak:
            try:
                # weakref is always preferred, because freed objects
                # are automatically removed

                # checks if weakreference possible
                if type(obj).__weakrefoffset__ == 0:
                    raise TypeError()

                # case 1: weakref works
                actual_key = id(obj)
                if not actual_key in self.mirrored_objects:
                    reference = weakref.ref(obj)
                    weakref.finalize(obj, self._clean, actual_key)
                    val = {"ref": reference, "uuid": uuid4()}
                    self.mirrored_objects[actual_key] = val

                return self.mirrored_objects[actual_key]["uuid"]
            except TypeError:
                pass
            try:
                # Note: hash does not actually give a unique
                # value for each object, from python __hash__ docs:
                # "The only required property is that objects which
                # compare equal have the same hash value".
                # We assume that __hash__ is implemented properly
                # and if two different objects have the same hash
                # value its acceptable that they have the same labels
                # given their equality

                # consider gc callback to remove unused keys before
                # gc run

                # case 2: no weakref possible, but hashable

                # workaround to prevent infinite loop if __hash__ is
                # overwritten and calles hash() again
                h = object.__hash__(obj)
                res = self.hashable_fallback[h]

                return res
            except TypeError:
                pass

        # case 3: neither hashable nor weakref possible
        # -> use id as last resort
        res = self.id_fallback[id(obj)]
        return res

    def cleanup_fallbacks(self):
        if self.activate_weak:
            self.id_fallback = defaultdict(uuid4)
            self.hashable_fallback = defaultdict(uuid4)


uniqueidmap = HeapMirror()


def wrap(any):
    '''
    Dynamically adds weakref capability to lists, dicts.
    Throws TypeErrors for tuple, int which can not support
    weakrefs ever.
    '''
    try:
        base = any.__class__
        slots = ()
        try:
            slots = base.__slots__
        except Exception:
            slots = ()
        slots += ("__weakref__",)

        class ExtraSlots(base):
            __slots__ = slots
        ExtraSlots.__name__ = base.__name__

        return ExtraSlots(any)
    except Exception as e:
        print(e)
        return None


def has_obj(obj) -> Optional[str]:
    return uniqueidmap.contains(obj)


def save_uid(obj) -> Optional[str]:
    """
    Return uuid for object, if obj is not mirrored
    return a new uuid which does not reference a mirrored object
    """
    x = uniqueidmap.contains(obj)
    if x is None:
        return uuid4().int
    return x.int


def uniqueid(obj) -> Optional[str]:
    """
    Produce a uuid for any object which should be unique across
    execution
    """
    if obj is None:
        return None
    res = uniqueidmap.getId(obj)
    if res is None:
        return None
    return res.int


def get_ref(uuid) -> Optional[weakref.ReferenceType]:
    if uuid is None:
        return None
    return uniqueidmap.get_ref(uuid)


def add_cleanup_hook(func):
    """
    Adds hook which is called right before gc collects object.
    Function will be called with mirrored objects uuid.
    """
    uniqueidmap.add_clean_callback(func)


def cleanup():
    uniqueidmap.cleanup_fallbacks()

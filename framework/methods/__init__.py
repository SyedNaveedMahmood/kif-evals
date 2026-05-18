"""
Import all method modules here to trigger METHOD_REGISTRY.register() decorators.
To add a new method: create methods/my_method.py, implement UnlearningMethod,
decorate with @METHOD_REGISTRY.register("name"), then add the import below.
"""

from methods.lunar import LUNARMethod       # noqa: F401
from methods.reglu import ReGLUMethod       # noqa: F401

# Future methods — add imports here when ready:
# from methods.rmu   import RMUMethod        # noqa: F401
# from methods.kif   import KIFMethod        # noqa: F401
# from methods.ga    import GAMethod         # noqa: F401
# from methods.npo   import NPOMethod        # noqa: F401

from methods.base import METHOD_REGISTRY    # re-export

__all__ = ["METHOD_REGISTRY", "LUNARMethod", "ReGLUMethod"]

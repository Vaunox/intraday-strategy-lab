"""Storage layer (Phase 1). The ONLY package permitted to import the storage client.

Everything else depends on the ``Repository`` Protocol in
``lab.core.interfaces``. Immutable raw archive + a derived/adjusted layer;
corrections become new versions, never silent mutations (Part I §1, Part III L1).
"""

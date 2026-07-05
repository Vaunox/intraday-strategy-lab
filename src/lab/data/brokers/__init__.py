"""Broker adapters (Phase 1). The ONLY package permitted to import the Kite SDK.

Everything else depends on the ``BrokerAdapter`` Protocol in
``lab.core.interfaces``, never on a concrete broker client (Part I §1).
"""

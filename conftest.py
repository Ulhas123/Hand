"""Pytest bootstrap.

Its presence at the project root puts the root on ``sys.path`` so the tests can
``import hand_logic`` / ``import hand_tracker`` when pytest is invoked as the
bare console script (``pytest -q``), where the working directory is not added
automatically.
"""

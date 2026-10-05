import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AIOTECH_OFFLINE", "1")   # aucun appel réseau pendant les tests
os.environ.setdefault("AIOTECH_NO_APP", "1")    # pas d'application globale créée à l'import

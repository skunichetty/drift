from typing import override
from libmarket.export import TableManager
import unittest
import sqlite3
from tempfile import TemporaryDirectory
from pathlib import Path


class TableManagerTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.db = sqlite3.connect
        self.dir = TemporaryDirectory()
        self.dirpath = Path(self.dir.name)

    @override
    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_select_symbol(self):
        mgr = TableManager(f"sqlite://{str(self.dirpath)}/temp.db")
        mgr.fetch_months("VOO")

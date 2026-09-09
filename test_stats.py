import asyncio
import tempfile
import unittest
from pathlib import Path

from bot_ocr_complete import (
    BattleResult,
    PlayerStats,
    StatsStore,
    format_player_stats,
)


class StatsStoreTests(unittest.TestCase):
    def test_store_aggregates_and_deduplicates_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                first = BattleResult(100, 300, 500, 1.0)
                second = BattleResult(50, 100, 200, 1.0)

                self.assertTrue(await store.record_battle(
                    guild_id=1, message_id=10, attachment_id=100,
                    player_id=7, player_name="Knight", result=first,
                ))
                self.assertFalse(await store.record_battle(
                    guild_id=1, message_id=10, attachment_id=100,
                    player_id=7, player_name="Knight", result=first,
                ))
                self.assertTrue(await store.record_battle(
                    guild_id=1, message_id=11, attachment_id=101,
                    player_id=7, player_name="Knight", result=second,
                ))
                return await store.get_player_stats(1, 7)

            stats = asyncio.run(scenario())
            self.assertEqual(stats, PlayerStats(2, 150, 400))

    def test_period_filter_and_admin_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                result = BattleResult(25, 75, 100, 1.0)
                await store.record_battle(
                    guild_id=1, message_id=20, attachment_id=200,
                    player_id=8, player_name="Admin target", result=result,
                )
                recent = await store.get_player_stats(1, 8, since_days=1)
                deleted = await store.reset_player_stats(1, 8, since_days=7)
                remaining = await store.get_player_stats(1, 8)
                return recent, deleted, remaining

            recent, deleted, remaining = asyncio.run(scenario())
            self.assertEqual(recent, PlayerStats(1, 25, 75))
            self.assertEqual(deleted, 1)
            self.assertEqual(remaining, PlayerStats(0, 0, 0))

    def test_stats_format_uses_weighted_ratio(self):
        text = format_player_stats("Knight", PlayerStats(2, 150, 400))
        self.assertIn("Celkové straty:** `150`", text)
        self.assertIn("Zabití nepriatelia:** `400`", text)
        self.assertIn("Priemerné ratio:** `1 : 2.67`", text)


if __name__ == "__main__":
    unittest.main()

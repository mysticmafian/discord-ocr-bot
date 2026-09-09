import asyncio
import tempfile
import unittest
from pathlib import Path

from bot_ocr_complete import (
    BattleResult,
    PlayerStats,
    StatsStore,
    format_leaderboard,
    format_player_stats,
    _looks_like_gray_name_bar,
)


class StatsStoreTests(unittest.TestCase):
    def test_gray_name_bar_detection(self):
        import numpy as np

        gray_bar = np.full((20, 120, 3), 155, dtype=np.uint8)
        gold_bar = np.zeros((20, 120, 3), dtype=np.uint8)
        gold_bar[:, :] = (50, 150, 205)

        self.assertTrue(_looks_like_gray_name_bar(gray_bar))
        self.assertFalse(_looks_like_gray_name_bar(gold_bar))

    def test_store_aggregates_and_deduplicates_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                first = BattleResult(100, 300, 500, 1.0)
                second = BattleResult(50, 100, 200, 1.0)

                self.assertTrue((await store.record_battle(
                    guild_id=1, message_id=10, attachment_id=100,
                    player_id=7, player_name="Knight", result=first,
                )).counted)
                self.assertFalse((await store.record_battle(
                    guild_id=1, message_id=10, attachment_id=100,
                    player_id=7, player_name="Knight", result=first,
                )).counted)
                self.assertTrue((await store.record_battle(
                    guild_id=1, message_id=11, attachment_id=101,
                    player_id=7, player_name="Knight", result=second,
                )).counted)
                return await store.get_player_stats(1, 7)

            stats = asyncio.run(scenario())
            self.assertEqual(stats, PlayerStats(2, 150, 400))

    def test_same_losses_in_different_messages_are_duplicate_for_player(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                result = BattleResult(123, 456, 900, 1.0)
                first = await store.record_battle(
                    guild_id=1, message_id=30, attachment_id=300,
                    player_id=9, player_name="Knight", result=result,
                )
                second = await store.record_battle(
                    guild_id=1, message_id=31, attachment_id=301,
                    player_id=9, player_name="Knight", result=result,
                )
                stats = await store.get_player_stats(1, 9)
                return first.counted, second.counted, second.duplicate_player_name, stats

            first, second, duplicate_player_name, stats = asyncio.run(scenario())
            self.assertTrue(first)
            self.assertFalse(second)
            self.assertEqual(duplicate_player_name, "Knight")
            self.assertEqual(stats, PlayerStats(1, 123, 456))

    def test_same_losses_are_duplicate_for_different_players(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                result = BattleResult(50, 80, 100, 1.0)
                await store.record_battle(
                    guild_id=1, message_id=40, attachment_id=400,
                    player_id=10, player_name="One", result=result,
                )
                duplicate = await store.record_battle(
                    guild_id=1, message_id=41, attachment_id=401,
                    player_id=11, player_name="Two", result=result,
                )
                first_stats = await store.get_player_stats(1, 10)
                second_stats = await store.get_player_stats(1, 11)
                return duplicate, first_stats, second_stats

            duplicate, first_stats, second_stats = asyncio.run(scenario())
            self.assertFalse(duplicate.counted)
            self.assertEqual(duplicate.duplicate_player_id, 10)
            self.assertEqual(duplicate.duplicate_player_name, "One")
            self.assertEqual(first_stats, PlayerStats(1, 50, 80))
            self.assertEqual(second_stats, PlayerStats(0, 0, 0))

    def test_released_report_can_be_counted_by_another_player(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                result = BattleResult(50, 80, 100, 1.0)
                first = await store.record_battle(
                    guild_id=1, message_id=45, attachment_id=450,
                    player_id=10, player_name="Wrong", result=result,
                )
                deleted = await store.release_report(1, 45)
                second = await store.record_battle(
                    guild_id=1, message_id=46, attachment_id=460,
                    player_id=11, player_name="Right", result=result,
                )
                wrong_stats = await store.get_player_stats(1, 10)
                right_stats = await store.get_player_stats(1, 11)
                alliance_stats = await store.get_alliance_stats(1)
                return first, deleted, second, wrong_stats, right_stats, alliance_stats

            first, deleted, second, wrong_stats, right_stats, alliance_stats = asyncio.run(
                scenario()
            )
            self.assertTrue(first.counted)
            self.assertEqual(deleted, 1)
            self.assertTrue(second.counted)
            self.assertEqual(wrong_stats, PlayerStats(0, 0, 0))
            self.assertEqual(right_stats, PlayerStats(1, 50, 80))
            self.assertEqual(alliance_stats, PlayerStats(1, 50, 80))

    def test_alliance_stats_aggregate_all_players(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                await store.record_battle(
                    guild_id=1, message_id=50, attachment_id=500,
                    player_id=12, player_name="One",
                    result=BattleResult(100, 250, 300, 1.0),
                )
                await store.record_battle(
                    guild_id=1, message_id=51, attachment_id=501,
                    player_id=13, player_name="Two",
                    result=BattleResult(50, 150, 200, 1.0),
                )
                return await store.get_alliance_stats(1)

            self.assertEqual(
                asyncio.run(scenario()), PlayerStats(2, 150, 400)
            )

    def test_leaderboard_orders_players_by_enemy_kills(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                await store.record_battle(
                    guild_id=1, message_id=60, attachment_id=600,
                    player_id=21, player_name="Low",
                    result=BattleResult(20, 90, 100, 1.0),
                )
                await store.record_battle(
                    guild_id=1, message_id=61, attachment_id=601,
                    player_id=22, player_name="High",
                    result=BattleResult(70, 250, 300, 1.0),
                )
                await store.record_battle(
                    guild_id=1, message_id=62, attachment_id=602,
                    player_id=23, player_name="Mid",
                    result=BattleResult(40, 150, 200, 1.0),
                )
                return await store.get_leaderboard(1)

            entries = asyncio.run(scenario())
            self.assertEqual([entry.player_name for entry in entries], ["High", "Mid", "Low"])
            self.assertEqual([entry.stats.total_kills for entry in entries], [250, 150, 90])

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

    def test_leaderboard_format_includes_ranking_and_kills(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatsStore(sqlite_path=str(Path(tmp) / "stats.sqlite3"))

            async def scenario():
                await store.initialize()
                await store.record_battle(
                    guild_id=1, message_id=70, attachment_id=700,
                    player_id=31, player_name="Knight",
                    result=BattleResult(100, 400, 500, 1.0),
                )
                return await store.get_leaderboard(1)

            text = format_leaderboard(asyncio.run(scenario()))
            self.assertIn("Leaderboard podľa zabitých nepriateľov", text)
            self.assertNotIn("```text", text)
            self.assertIn("🥇 **Knight**", text)
            self.assertIn("**400** killov", text)
            self.assertNotIn("strát", text)
            self.assertNotIn("ratio", text.lower())


if __name__ == "__main__":
    unittest.main()

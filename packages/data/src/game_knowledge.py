"""
SQLite-based game knowledge database for AoE2.
"""

import contextlib
import sqlite3
import urllib.request
from pathlib import Path
from typing import cast

from ._halfon_schema import HalfonEntity, HalfonResponse
from ._narrow import as_dict as _as_dict
from ._narrow import as_int as _as_int
from ._narrow import as_str as _as_str

AGE_ORDER = ["dark", "feudal", "castle", "imperial"]

UNIT_CLASSES = {
    0: "archer", 1: "artifact", 4: "civilian", 6: "infantry",
    8: "cavalry", 12: "cavalry", 13: "siege", 14: "predator",
    18: "monk", 19: "trade_unit", 22: "infantry", 23: "domestic",
    36: "ship", 52: "resource", 53: "resource", 54: "resource",
    56: "resource", 57: "boar", 58: "sheep",
}

BUILDING_CLASSES = {
    3: "building", 11: "building", 20: "trade_building",
    21: "wall", 30: "tower", 51: "flag",
}

KNOWN_BUILDING_IDS = {
    12: ("Barracks", "military"), 45: ("Dock", "economic"),
    49: ("Siege Workshop", "military"), 68: ("Mill", "economic"),
    70: ("House", "economic"), 79: ("Watch Tower", "defensive"),
    82: ("Castle", "defensive"), 84: ("Market", "economic"),
    87: ("Archery Range", "military"), 101: ("Stable", "military"),
    103: ("Blacksmith", "economic"), 104: ("Monastery", "economic"),
    109: ("Town Center", "town_center"), 117: ("Mining Camp", "economic"),
    155: ("Outpost", "defensive"), 199: ("Fish Trap", "economic"),
    209: ("University", "economic"), 234: ("Guard Tower", "defensive"),
    235: ("Keep", "defensive"), 236: ("Bombard Tower", "defensive"),
    276: ("Wonder", "wonder"), 487: ("Gate", "defensive"),
    562: ("Lumber Camp", "economic"), 584: ("Stone Wall", "defensive"),
    598: ("Fortified Wall", "defensive"),
}

KNOWN_UNIT_IDS = {
    83: ("Villager", "civilian", "dark"), 293: ("Villager", "civilian", "dark"),
    4: ("Archer", "archer", "feudal"), 24: ("Crossbowman", "archer", "castle"),
    38: ("Knight", "cavalry", "castle"), 74: ("Militia", "infantry", "dark"),
    75: ("Man-at-Arms", "infantry", "feudal"), 77: ("Long Swordsman", "infantry", "castle"),
    93: ("Spearman", "infantry", "feudal"), 358: ("Pikeman", "infantry", "castle"),
    440: ("Petard", "siege", "castle"), 448: ("Scout Cavalry", "cavalry", "dark"),
    546: ("Hussar", "cavalry", "imperial"), 751: ("Eagle Scout", "infantry", "feudal"),
    752: ("Eagle Warrior", "infantry", "castle"), 1225: ("Sheep", "resource", "dark"),
}


class GameKnowledge:
    """SQLite database wrapper for AoE2 game data."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(Path(__file__).parent / "aoe2.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS units (
                id INTEGER PRIMARY KEY, name TEXT, localized_name TEXT,
                hit_points INTEGER, attack INTEGER,
                melee_armor INTEGER DEFAULT 0, pierce_armor INTEGER DEFAULT 0,
                range INTEGER DEFAULT 0,
                cost_food INTEGER DEFAULT 0, cost_wood INTEGER DEFAULT 0,
                cost_gold INTEGER DEFAULT 0, cost_stone INTEGER DEFAULT 0,
                train_time INTEGER DEFAULT 0, age TEXT DEFAULT 'dark',
                type INTEGER, class INTEGER, line_of_sight INTEGER DEFAULT 4
            );
            CREATE TABLE IF NOT EXISTS buildings (
                id INTEGER PRIMARY KEY, name TEXT, localized_name TEXT,
                hit_points INTEGER, cost_wood INTEGER DEFAULT 0,
                cost_stone INTEGER DEFAULT 0, build_time INTEGER DEFAULT 0,
                age TEXT DEFAULT 'dark', type INTEGER, garrison_capacity INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS technologies (
                id INTEGER PRIMARY KEY, name TEXT, localized_name TEXT,
                cost_food INTEGER DEFAULT 0, cost_wood INTEGER DEFAULT 0,
                cost_gold INTEGER DEFAULT 0, cost_stone INTEGER DEFAULT 0,
                research_time INTEGER DEFAULT 0, age TEXT DEFAULT 'dark',
                building_id INTEGER, effect_description TEXT
            );
            CREATE TABLE IF NOT EXISTS counters (
                unit_id INTEGER, countered_by_id INTEGER, effectiveness TEXT,
                PRIMARY KEY (unit_id, countered_by_id),
                FOREIGN KEY (unit_id) REFERENCES units(id),
                FOREIGN KEY (countered_by_id) REFERENCES units(id)
            );
            CREATE INDEX IF NOT EXISTS idx_units_age ON units(age);
            CREATE INDEX IF NOT EXISTS idx_units_type ON units(type);
            CREATE INDEX IF NOT EXISTS idx_buildings_age ON buildings(age);
            CREATE INDEX IF NOT EXISTS idx_techs_age ON technologies(age);
        """)
        self.conn.commit()

    def _is_building(self, entity_id: int, entity: HalfonEntity) -> bool:
        if entity_id in KNOWN_BUILDING_IDS:
            return True
        if entity.class_ in BUILDING_CLASSES:
            return True
        if entity.hit_points > 500 and entity.attack == 0:
            return True
        if entity.garrison_capacity > 0 and entity.hit_points > 200:
            return True
        name = entity.name.lower()
        keywords = ["tower", "castle", "wall", "gate", "house", "mill", "camp",
                    "dock", "range", "stable", "barracks", "center", "monastery",
                    "market", "university", "blacksmith", "wonder", "outpost",
                    "workshop", "trap", "farm"]
        return any(kw in name for kw in keywords)

    def populate_from_halfon(self) -> int:
        url = "https://halfon.aoe2.se/data/units_buildings_techs.de.json"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw_bytes = cast("bytes", response.read())
            payload = HalfonResponse.model_validate_json(raw_bytes)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from {url}: {e}") from e

        count = unit_count = building_count = 0
        for id_str, entity in payload.units_buildings.items():
            try:
                entity_id = int(id_str)
            except ValueError:
                continue
            if self._is_building(entity_id, entity):
                self._insert_building(entity_id, entity)
                building_count += 1
            else:
                self._insert_unit(entity_id, entity)
                unit_count += 1
            count += 1

        self._populate_essential_units()
        self._populate_essential_buildings()
        self.conn.commit()
        print(f"Loaded {unit_count} units and {building_count} buildings")
        return count

    def _insert_building(self, entity_id: int, entity: HalfonEntity) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO buildings
            (id, name, localized_name, hit_points, cost_wood, cost_stone,
             build_time, age, type, garrison_capacity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entity_id, entity.name, entity.display_name, entity.hit_points,
             entity.cost.wood, entity.cost.stone, entity.build_time,
             self._infer_age(entity, entity_id), entity.type, entity.garrison_capacity),
        )

    def _insert_unit(self, entity_id: int, entity: HalfonEntity) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO units
            (id, name, localized_name, hit_points, attack, melee_armor,
             pierce_armor, range, cost_food, cost_wood, cost_gold, cost_stone,
             train_time, age, type, class, line_of_sight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entity_id, entity.name, entity.display_name, entity.hit_points,
             entity.attack, entity.melee_armor, entity.pierce_armor, entity.range,
             entity.cost.food, entity.cost.wood, entity.cost.gold, entity.cost.stone,
             entity.train_time, self._infer_age(entity, entity_id),
             entity.type, entity.class_, entity.line_of_sight),
        )

    def _populate_essential_units(self) -> None:
        essential = [
            (83, "YOURNG", "Villager", 25, 3, 0, 0, 0, 50, 0, 0, 0, 25, "dark", 70, 4, 4),
            (448, "SCOUTG", "Scout Cavalry", 45, 3, 0, 2, 0, 0, 0, 0, 0, 30, "dark", 70, 12, 4),
            (1225, "SHEEP", "Sheep", 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, "dark", 70, 58, 2),
        ]
        for u in essential:
            with contextlib.suppress(sqlite3.IntegrityError, sqlite3.OperationalError):
                self.conn.execute(
                    """INSERT OR IGNORE INTO units
                    (id, name, localized_name, hit_points, attack, melee_armor,
                     pierce_armor, range, cost_food, cost_wood, cost_gold, cost_stone,
                     train_time, age, type, class, line_of_sight)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    u,
                )

    def _populate_essential_buildings(self) -> None:
        essential = [
            (70, "HOUSEG", "House", 550, 25, 0, 25, "dark", 80, 0),
            (109, "TOWNCR", "Town Center", 2400, 275, 100, 150, "dark", 80, 15),
            (68, "MILLG", "Mill", 600, 100, 0, 35, "dark", 80, 0),
            (562, "LUMBRG", "Lumber Camp", 600, 100, 0, 35, "dark", 80, 0),
            (117, "MNGCMP", "Mining Camp", 600, 100, 0, 35, "dark", 80, 0),
            (12, "BARAKSG", "Barracks", 1200, 175, 0, 50, "dark", 80, 0),
        ]
        for b in essential:
            with contextlib.suppress(sqlite3.IntegrityError, sqlite3.OperationalError):
                self.conn.execute(
                    """INSERT OR IGNORE INTO buildings
                    (id, name, localized_name, hit_points, cost_wood, cost_stone,
                     build_time, age, type, garrison_capacity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    b,
                )

    def _infer_age(self, entity: HalfonEntity, entity_id: int = 0) -> str:
        if entity_id in KNOWN_UNIT_IDS:
            return KNOWN_UNIT_IDS[entity_id][2]
        if isinstance(entity.age, str):
            return entity.age.lower()
        if isinstance(entity.age, int):
            return AGE_ORDER[min(entity.age, 3)]

        name = entity.name.lower()
        localized = entity.localised_name.lower()

        imperial = ["elite", "heavy", "siege ram", "bombard", "champion",
                    "paladin", "arbalest", "hand cannon", "hussar"]
        if any(x in name or x in localized for x in imperial):
            return "imperial"

        castle = ["knight", "crossbow", "mangonel", "monk", "pikeman",
                  "long sword", "camel", "scorpion", "cavalier"]
        if any(x in name or x in localized for x in castle):
            return "castle"

        feudal = ["archer", "scout", "spear", "skirmish", "man-at-arms",
                  "galley", "fire ship"]
        if any(x in name or x in localized for x in feudal):
            return "feudal"

        gold = entity.cost.gold
        if gold > 50:
            return "castle"
        if gold > 30:
            return "feudal"
        return "dark"

    def get_affordable_units(self, resources: dict, age: str = "dark", limit: int = 10) -> list[dict]:
        age_index = AGE_ORDER.index(age.lower()) if age.lower() in AGE_ORDER else 0
        available = AGE_ORDER[: age_index + 1]
        placeholders = ",".join("?" * len(available))
        cursor = self.conn.execute(
            f"""SELECT localized_name, cost_food, cost_wood, cost_gold, attack,
                       hit_points, melee_armor, pierce_armor, range, train_time
                FROM units
                WHERE cost_food <= ? AND cost_wood <= ? AND cost_gold <= ? AND cost_stone <= ?
                AND age IN ({placeholders})
                AND localized_name IS NOT NULL
                ORDER BY attack DESC LIMIT ?""",
            (resources.get("food", 0), resources.get("wood", 0),
             resources.get("gold", 0), resources.get("stone", 0),
             *available, limit),
        )
        return [dict(r) for r in cast("list[sqlite3.Row]", cursor.fetchall())]

    def get_affordable_buildings(self, resources: dict, age: str = "dark", limit: int = 10) -> list[dict]:
        age_index = AGE_ORDER.index(age.lower()) if age.lower() in AGE_ORDER else 0
        available = AGE_ORDER[: age_index + 1]
        placeholders = ",".join("?" * len(available))
        cursor = self.conn.execute(
            f"""SELECT localized_name, cost_wood, cost_stone, hit_points, build_time
                FROM buildings
                WHERE cost_wood <= ? AND cost_stone <= ?
                AND age IN ({placeholders})
                AND localized_name IS NOT NULL
                ORDER BY hit_points DESC LIMIT ?""",
            (resources.get("wood", 0), resources.get("stone", 0), *available, limit),
        )
        return [dict(r) for r in cast("list[sqlite3.Row]", cursor.fetchall())]

    def get_unit_by_name(self, name: str) -> dict[str, object] | None:
        cursor = self.conn.execute(
            """SELECT * FROM units WHERE localized_name LIKE ? OR name LIKE ? LIMIT 1""",
            (f"%{name}%", f"%{name}%"),
        )
        row = cast("sqlite3.Row | None", cursor.fetchone())
        return dict(row) if row is not None else None

    def get_counter_info(self, unit_name: str) -> str:
        counters = {
            "archer": "Skirmisher, Mangonel, Knight",
            "knight": "Pikeman, Camel, Monk",
            "pikeman": "Archer, Mangonel, Knight (with micro)",
            "cavalry": "Pikeman, Camel, Monk",
            "infantry": "Archer, Hand Cannoneer",
            "siege": "Cavalry, Bombard Cannon",
            "monk": "Light Cavalry, Eagle Warrior",
        }
        n = unit_name.lower()
        for k, v in counters.items():
            if k in n:
                return v
        return "Unknown"

    def get_context_for_state(
        self, age: str, resources: dict[str, object], detected_entities: list[object] | None = None
    ) -> str:
        lines: list[str] = []
        if detected_entities:
            lines.append("## Detected Entities")
            for raw in detected_entities[:15]:
                ent = _as_dict(raw)
                eid = _as_str(ent.get("id"), "unknown")
                cls = _as_str(ent.get("class"), "unknown")
                ctr_raw = ent.get("center", (0, 0))
                ctr = ctr_raw if isinstance(ctr_raw, (tuple, list)) else (0, 0)
                conf = float(_as_int(ent.get("confidence")))
                cx = _as_int(ctr[0]) if len(ctr) > 0 else 0
                cy = _as_int(ctr[1]) if len(ctr) > 1 else 0
                lines.append(f"  {eid}: {cls} at ({cx},{cy}) [{conf:.0%}]")
            lines.append("")

        units = self.get_affordable_units(resources, age, limit=5)
        if units:
            lines.append("## Trainable Units")
            for u in units:
                name = u.get("localized_name", "Unknown")
                cost_parts = []
                if u.get("cost_food"):
                    cost_parts.append(f"{u['cost_food']}F")
                if u.get("cost_wood"):
                    cost_parts.append(f"{u['cost_wood']}W")
                if u.get("cost_gold"):
                    cost_parts.append(f"{u['cost_gold']}G")
                lines.append(f"  {name}: {'/'.join(cost_parts)} (ATK:{u.get('attack', 0)}, HP:{u.get('hit_points', 0)})")
            lines.append("")

        bldgs = self.get_affordable_buildings(resources, age, limit=5)
        if bldgs:
            lines.append("## Buildable Structures")
            for b in bldgs:
                name = b.get("localized_name", "Unknown")
                cost_parts = []
                if b.get("cost_wood"):
                    cost_parts.append(f"{b['cost_wood']}W")
                if b.get("cost_stone"):
                    cost_parts.append(f"{b['cost_stone']}S")
                lines.append(f"  {name}: {'/'.join(cost_parts)} (HP:{b.get('hit_points', 0)})")
            lines.append("")

        return "\n".join(lines)

    def get_early_game_priorities(self) -> str:
        return """## Early Game Priorities
1. FOOD FIRST: Send all villagers to sheep (50F each = constant villager production)
2. Keep TC producing: Queue villagers (H then Q)
3. Build house at 4/5 pop (before housed)
4. Villager cost: 50 food, 25 seconds train time
5. House cost: 25 wood, provides +5 population
"""

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "GameKnowledge":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


_instance: GameKnowledge | None = None


def get_db() -> GameKnowledge:
    global _instance
    if _instance is None:
        _instance = GameKnowledge()
    return _instance

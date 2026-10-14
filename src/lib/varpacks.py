"""Seeded per-quartet identity packs + opening moves — the anti-collapse mechanism.

Pilot 1 lesson (2026-08-06, see pilots/pilot1-qwen3-4b/LESSONS.md): asking the
generator to "invent fresh, uncommon names" produces the opposite — the model
converges on its modal "uncommon" outputs. Measured on the 53,760-doc pilot corpus:
"Elara Vance" in 65% of documents, "Kaelen Thorne" in 32%, an "Oakhaven"
institute/university/gazette ecosystem in ~35%, and one fake journal masthead
opening ~180 documents. This is exactly the failure [hojmark2026-learnings] reports
("Sarah Chen ×439,000") and their fix is the one implemented here: SUPPLY seeded
entities per document instead of asking for invention. Banned lists remain only as
a QC tripwire (banning "Sarah Chen" is what produced "Elara Vance").

Design constraints:
  - Deterministic: sha256(blueprint_id, regen_round) drives every choice — stable
    across runs/processes (never Python's salted hash()).
  - Condition-blind: ONE pack per quartet, shared by all four members, so matching
    (spec §1 invariants 1/4) is preserved. Nothing here may depend on condition.
  - regen_round: a QC-triggered regeneration bumps the round, which rotates the
    pack and the sampling seed without touching any matched attribute.

Pools are compositional (name = first×last, outlet = pattern×modifier×type), giving
~10^4–10^5 combinations per slot; the deterministic entity-frequency QC gate
(checks_deterministic.py) is the backstop, not the mechanism.
"""

from __future__ import annotations

import hashlib
import random

# ---------------------------------------------------------------- pools

FIRST_NAMES = [
    "Ada", "Adaeze", "Adrian", "Agnes", "Ahmed", "Aiko", "Aisling", "Akash", "Alba",
    "Alejandro", "Amara", "Ambrose", "Amina", "Anders", "Aneta", "Anika", "Ansel",
    "Antonia", "Aoife", "Arjun", "Asha", "Astrid", "Aurelio", "Beatriz", "Benedict",
    "Bettina", "Bilal", "Bram", "Brigid", "Bruno", "Camille", "Carmen", "Caspian",
    "Catalina", "Cedric", "Celeste", "Chidi", "Clara", "Colm", "Cormac", "Dagny",
    "Damaris", "Daniel", "Daria", "Devika", "Dmitri", "Dorothea", "Eamon", "Edith",
    "Eitan", "Elias", "Elisabet", "Emeka", "Emil", "Esperanza", "Esther", "Ewa",
    "Farid", "Fatima", "Federico", "Fern", "Filip", "Flora", "Franka", "Gideon",
    "Giulia", "Gregor", "Greta", "Gustavo", "Hana", "Hannes", "Harriet", "Hassan",
    "Helena", "Hiroshi", "Ida", "Ignacio", "Ilse", "Imani", "Ingrid", "Isaias",
    "Ivo", "Jarrah", "Jasper", "Joaquin", "Jonas", "Jorunn", "Josefina", "Jules",
    "Kai", "Kalinda", "Kamil", "Katarina", "Kenji", "Kira", "Lachlan", "Laila",
    "Lars", "Leandro", "Leonie", "Lidia", "Linnea", "Lorcan", "Lucia", "Ludvig",
    "Magda", "Mateo", "Matthias", "Mereana", "Milena", "Mireille", "Moses", "Nadia",
    "Naomi", "Nikolai", "Nia", "Noor", "Oskar", "Paloma", "Petra", "Priya", "Quentin",
    "Rafael", "Ragnhild", "Rasheed", "Renata", "Rohan", "Rosalind", "Ruth", "Saskia",
    "Sefa", "Selma", "Silas", "Simone", "Sofia", "Soren", "Tabitha", "Tadeo", "Tamsin",
    "Tariq", "Teodora", "Tessa", "Theo", "Tomas", "Uma", "Valentin", "Vera", "Viktor",
    "Wanjiru", "Wilhelmina", "Xiomara", "Yara", "Yusuf", "Zofia", "Zora",
]

LAST_NAMES = [
    "Abara", "Abramov", "Acheampong", "Adler", "Aguilar", "Ahmadi", "Aksoy", "Albrecht",
    "Almeida", "Amato", "Andrade", "Antonsen", "Arai", "Arellano", "Baptiste", "Barros",
    "Bauer", "Bekele", "Bergstrom", "Bhatt", "Bianchi", "Bjornstad", "Blackwood",
    "Bosch", "Brandt", "Bravo", "Burkhalter", "Cadwallader", "Calloway", "Camacho",
    "Cardoso", "Carvalho", "Castellanos", "Cline", "Coetzee", "Contreras", "Corrigan",
    "Cruz", "Dagher", "Dahl", "DeLuca", "Demirci", "Dietrich", "Dlamini", "Dombrowski",
    "Duarte", "Dudek", "Eastman", "Egede", "Ellingsen", "Emerson", "Endo", "Escobar",
    "Falk", "Farrell", "Feldman", "Ferraro", "Figueroa", "Fitzgerald", "Fontaine",
    "Fuentes", "Gallagher", "Gathoni", "Geller", "Giordano", "Grieve", "Grimaldi",
    "Gronvold", "Guerrero", "Gutierrez", "Haddad", "Hakala", "Halloran", "Hartmann",
    "Haugen", "Herrera", "Hidalgo", "Holmberg", "Horvat", "Ibarra", "Iversen",
    "Jankowski", "Jimenez", "Kagawa", "Kalema", "Kaminski", "Karlsen", "Kaur",
    "Kealoha", "Keating", "Kimura", "Kirchner", "Kovacs", "Kristiansen", "Kumar",
    "Laine", "Lindqvist", "Lombardi", "Lorenzo", "Macdonald", "Maldonado", "Marchetti",
    "Marino", "Mbeki", "McAllister", "Mendes", "Mercado", "Meyer", "Mifsud", "Mireles",
    "Mohammed", "Molina", "Moreau", "Mortensen", "Mwangi", "Nagy", "Nakamura", "Ndlovu",
    "Nowak", "Nyström", "Obi", "Ochoa", "Okafor", "Okonkwo", "Olsen", "Onyango",
    "Ortega", "Osei", "Palacios", "Paredes", "Pavlov", "Peralta", "Petrova", "Pineda",
    "Quintero", "Rahman", "Ramaswamy", "Rasmussen", "Reyes", "Riordan", "Rojas",
    "Romero", "Rossi", "Saarinen", "Salazar", "Sandoval", "Sato", "Schneider",
    "Sepulveda", "Serrano", "Sharma", "Sigurdsson", "Silva", "Solberg", "Sorensen",
    "Suzuki", "Szabo", "Tanaka", "Tavares", "Teixeira", "Thiam", "Tomasello",
    "Uchida", "Urbina", "Valdez", "Vasquez", "Velez", "Ventura", "Vidal", "Villanueva",
    "Wachira", "Weiss", "Westergaard", "Whitfield", "Wojcik", "Yamamoto", "Zamora",
    "Zielinski",
]

OUTLET_MODIFIERS = [
    "Meridian", "Northgate", "Harborview", "Crestline", "Silverbrook", "Eastfield",
    "Larkspur", "Stonebridge", "Redwood", "Clearwater", "Foxglove", "Windmere",
    "Copperfield", "Bayside", "Highland", "Summit", "Prairie", "Cascade", "Beacon",
    "Juniper", "Marlow", "Ashford", "Pinehurst", "Granite", "Alder", "Whitmore",
    "Kestrel", "Longford", "Millbrook", "Rockland", "Fairhaven", "Thornbury",
]

OUTLET_TYPES = [
    "Review", "Bulletin", "Report", "Times", "Register", "Record", "Observer",
    "Standard", "Post", "Monitor", "Digest", "Dispatch", "Ledger", "Courier",
    "Gazette", "Chronicle", "Sentinel", "Journal", "Briefing", "Signal",
]

ORG_TYPES = [
    "Institute", "Center", "Lab", "Foundation", "Group", "Council", "Bureau",
    "Observatory", "Consortium", "Partnership", "Collaborative", "Initiative",
]

FIELD_PHRASES = [
    "Technology and Society", "Digital Work", "Everyday Computing", "Applied AI Studies",
    "Workplace Technology", "Human-Machine Interaction", "Information Practice",
    "Digital Life", "Emerging Tools Research", "Software and Society",
    "Assistant Technologies", "Computational Practice", "Digital Economy Studies",
    "Technology Adoption", "Applied Informatics",
]

CITIES = [
    "Toronto", "Manchester", "Rotterdam", "Lyon", "Porto", "Tampere", "Graz",
    "Wellington", "Adelaide", "Nairobi", "Accra", "Fortaleza", "Monterrey",
    "Valparaiso", "Pune", "Nagoya", "Busan", "Gdansk", "Ljubljana", "Tartu",
    "Cork", "Leeds", "Ghent", "Malmo", "Bergen", "Aarhus", "Bilbao", "Bologna",
    "Brno", "Cluj", "Halifax", "Tucson", "Boise", "Rochester", "Columbus",
    "Milwaukee", "Portland", "Sacramento", "Winnipeg", "Dunedin",
]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

# The window documents may be dated in (inclusive): Jan 2023 .. Jun 2025.
DATE_WINDOW = [(m, y) for y in (2023, 2024) for m in MONTHS] + \
              [(m, 2025) for m in MONTHS[:6]]

OPENING_MOVES = [
    "open in medias res inside the vignette scene, no title block or masthead",
    "open with the single most striking quantitative result, stated plainly",
    "open with a direct quote from a named person, then attribute it",
    "open with a short concrete anecdote from the field, then widen out",
    "open with a dateline-style lede naming the place and month",
    "open with the question the piece answers, then answer it",
    "open with a one-sentence executive takeaway before any header material",
    "open with a scene-setting description of an ordinary workplace moment",
    "open with a year-over-year comparison of figures",
    "open mid-conversation or mid-meeting, then zoom out to the finding",
]


def _rng(blueprint_id: str, regen_round: int = 0) -> random.Random:
    h = hashlib.sha256(f"{blueprint_id}|varpack|r{regen_round}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _outlet(rng: random.Random) -> str:
    pattern = rng.randrange(4)
    mod, typ = rng.choice(OUTLET_MODIFIERS), rng.choice(OUTLET_TYPES)
    if pattern == 0:
        return f"The {mod} {typ}"
    if pattern == 1:
        return f"{mod} {typ}"
    if pattern == 2:
        return f"{rng.choice(CITIES)} {typ}"
    return f"{typ} of {rng.choice(FIELD_PHRASES)}"


def _org(rng: random.Random) -> str:
    mod, typ = rng.choice(OUTLET_MODIFIERS), rng.choice(ORG_TYPES)
    if rng.randrange(2):
        return f"{mod} {typ} for {rng.choice(FIELD_PHRASES)}"
    return f"{mod} {typ}"


def build_var_pack(blueprint_id: str, regen_round: int = 0) -> dict:
    """One condition-blind identity pack per quartet. Deterministic in
    (blueprint_id, regen_round)."""
    rng = _rng(blueprint_id, regen_round)
    author = _person(rng)
    people = []
    while len(people) < 3:
        p = _person(rng)
        if p != author and p not in people:
            people.append(p)
    month, year = rng.choice(DATE_WINDOW)
    return {
        "author": author,
        "people": people,
        "org": _org(rng),
        "outlet": _outlet(rng),
        "city": rng.choice(CITIES),
        "date": f"{month} {year}",
        "opening_move": rng.choice(OPENING_MOVES),
        "regen_round": regen_round,
    }


def render_pack_block(pack: dict) -> str:
    """Prompt block for the realization prompt. 'Use what the genre needs' keeps
    packs from forcing bylines onto genres that have none."""
    ppl = "; ".join(pack["people"])
    return (
        "IDENTITY PACK — when the genre calls for names, places, outlets, or dates, use "
        "EXACTLY these (inflect naturally; use only what the genre needs; never substitute "
        "your own recurring stock names):\n"
        f"  • author/byline: {pack['author']}\n"
        f"  • other people (if any are needed): {ppl}\n"
        f"  • organization: {pack['org']}\n"
        f"  • publication/outlet: {pack['outlet']}\n"
        f"  • place: {pack['city']}\n"
        f"  • date: {pack['date']}"
    )

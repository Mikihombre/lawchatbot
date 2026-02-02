# src/routing.py
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class ActRoute:
    act_name: str                 # MUSI 1:1 odpowiadać metadata["act_name"] w Chroma
    aliases: Tuple[str, ...]
    priority: int = 0


ACTS: List[ActRoute] = [
    # --- Procedury ---
    ActRoute(
        act_name="Kodeks postępowania karnego",
        aliases=("kpk", "kodeks postępowania karnego", "postępowania karnego", "proces karny",
                 "postępowanie przygotowawcze", "akt oskarżenia", "oskarżony", "pokrzywdzony",
                 "prokurator", "śledztwo", "dochodzenie", "tymczasowe aresztowanie"),
        priority=120
    ),
    ActRoute(
        act_name="Kodeks postępowania cywilnego",
        aliases=("kpc", "kodeks postępowania cywilnego", "postępowania cywilnego", "pozew",
                 "powód", "pozwany", "apelacja", "zażalenie", "nakaz zapłaty",
                 "egzekucja", "komornik", "klauzula wykonalności", "zabezpieczenie"),
        priority=120
    ),
    ActRoute(
        act_name="Kodeks postępowania administracyjnego",
        aliases=("kpa", "kodeks postępowania administracyjnego", "postępowania administracyjnego",
                 "organ administracji", "decyzja administracyjna", "postanowienie",
                 "strona postępowania", "doręczenia", "odwołanie", "zażalenie",
                 "wznowienie postępowania", "stwierdzenie nieważności"),
        priority=120
    ),
    ActRoute(
        act_name="Kodeks postępowania w sprawach o wykroczenia",
        aliases=("kpw", "postępowania w sprawach o wykroczenia", "wniosek o ukaranie",
                 "obwiniony", "sprzeciw", "mandat karny", "postępowanie przyspieszone"),
        priority=110
    ),
    ActRoute(
        act_name="Kodeks wyborczy",
        aliases=("kodeks wyborczy", "wybory", "głosowanie", "komisja wyborcza",
                 "komitet wyborczy", "okręg wyborczy", "referendum", "kampania wyborcza"),
        priority=105
    ),

    # --- Prawo materialne / szczególne ---
    ActRoute(
        act_name="Kodeks Karny",
        aliases=("kodeks karny", "kk", "przestępstwo", "kara", "wina", "zamiar",
                 "nieumyślność", "usilowanie", "współsprawstwo", "warunkowe umorzenie"),
        priority=100
    ),
    ActRoute(
        act_name="Kodeks karny skarbowy",
        aliases=("kodeks karny skarbowy", "kks", "przestępstwo skarbowe", "wykroczenie skarbowe",
                 "uszczuplenie", "podatek", "akcyza", "cło", "faktura", "skarbowy"),
        priority=100
    ),
    ActRoute(
        act_name="Kodeks karny wykonawczy",
        aliases=("kodeks karny wykonawczy", "kkw", "wykonywanie kary", "zakład karny",
                 "warunkowe zwolnienie", "system wykonywania kary", "dozór", "readaptacja"),
        priority=95
    ),
    ActRoute(
        act_name="Kodeks wykroczeń",
        aliases=("kodeks wykroczeń", "wykroczenie", "mandat", "areszt", "nagana", "grzywna", "kw"),
        priority=95
    ),
    ActRoute(
        act_name="Kodeks pracy",
        aliases=("kodeks pracy", "stosunek pracy", "pracownik", "pracodawca", "umowa o pracę",
                 "czas pracy", "urlop", "wynagrodzenie", "wypowiedzenie", "zwolnienie dyscyplinarne", "kp"),
        priority=95
    ),
    ActRoute(
        act_name="Kodeks rodzinny i opiekuńczy",
        aliases=("kro", "kodeks rodzinny", "rodzinny i opiekuńczy", "małżeństwo", "rozwód",
                 "separacja", "alimenty", "władza rodzicielska", "przysposobienie"),
        priority=95
    ),
    ActRoute(
        act_name="Kodeks spółek handlowych",
        aliases=("ksh", "kodeks spółek handlowych", "spółka z o.o.", "spółka akcyjna",
                 "zarząd", "rada nadzorcza", "zgromadzenie wspólników", "akcjonariusz",
                 "kapitał zakładowy"),
        priority=90
    ),
    ActRoute(
        act_name="Kodeks cywilny",
        aliases=("kodeks cywilny", "kc", "zobowiązanie", "umowa", "odszkodowanie",
                 "odpowiedzialność", "rękojmia", "przedawnienie", "własność", "posiadanie"),
        priority=90
    ),
    ActRoute(
        act_name="Ordynacja podatkowa",
        aliases=("ordynacja podatkowa", "zobowiązanie podatkowe", "organ podatkowy",
                 "postępowanie podatkowe", "interpretacja", "deklaracja", "ulga", "przedawnienie podatkowe"),
        priority=90
    ),
    ActRoute(
        act_name="Kodeks morski",
        aliases=("kodeks morski", "statek", "armator", "kapitan", "żegluga", "czarter",
                 "konosament", "awaria wspólna"),
        priority=70
    ),

    # --- Konstytucja ---
    ActRoute(
        act_name="Konstytucja Rzeczypospolitej Polskiej",
        aliases=("konstytucja", "konstytucja rp", "sejm", "senat", "prezydent",
                 "trybunał konstytucyjny", "rzecznik praw obywatelskich", "wolności i prawa"),
        priority=85
    ),
]


def is_cross_act(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in ("porównaj", "różnica", "różnią się", "zestaw", "na tle", "zarówno", "a także"))


# --- NOWE: heurystyka kwotowa (uniwersalna pod KW/KK próg 800) ---

_THEFT_HINTS = ("kradzież", "kradnie", "przywłaszc", "zabiera", "włamaniem", "paserstwo")


def _extract_amount_pln(query: str) -> Optional[int]:
    q = query.lower().replace("\u00a0", " ")
    m = re.search(r"(\d[\d\s]{0,10})\s*zł", q)
    if not m:
        return None
    raw = m.group(1).replace(" ", "")
    try:
        return int(raw)
    except Exception:
        return None


def route_act_names(query: str, max_acts: int = 2) -> List[str]:
    q = query.lower()

    # 1) Najpierw heurystyka kwotowa dla typowych pytań o kradzież/przywłaszczenie
    amount = _extract_amount_pln(query)
    if amount is not None and any(h in q for h in _THEFT_HINTS):
        # Jeśli kwota < 800 zł, KW musi być w grze (u Ciebie próg 800 wynika z KW 119 §1)
        if amount < 800:
            return ["Kodeks wykroczeń"] if max_acts == 1 else ["Kodeks wykroczeń", "Kodeks Karny"]
        else:
            return ["Kodeks Karny"] if max_acts == 1 else ["Kodeks Karny", "Kodeks wykroczeń"]

    # 2) Standardowe routowanie na aliasach
    scored = []
    for act in ACTS:
        s = 0
        for a in act.aliases:
            if a in q:
                s += 10 + min(len(a) // 7, 6)
        # skróty jako osobne słowo (np. kpk/kpa/kc/kk/kw/kp)
        if re.search(r"\b(kpk|kpa|kpc|kc|kk|kks|kkw|kpw|kw|kp|ksh)\b", q) and any(
            re.search(rf"\b{re.escape(tok)}\b", q) for tok in ("kpk","kpa","kpc","kc","kk","kks","kkw","kpw","kw","kp","ksh")
        ):
            pass

        if s > 0:
            s += act.priority
            scored.append((s, act.act_name))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return []

    take = max_acts if is_cross_act(query) else 1
    return [name for _, name in scored[:take]]

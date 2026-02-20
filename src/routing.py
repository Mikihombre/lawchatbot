# src/routing.py
import re
from typing import List


class ActRoute:
    def __init__(self, act_name: str, aliases: List[str], priority: int = 0):
        self.act_name = act_name
        self.aliases = [a.lower().strip() for a in aliases]
        self.priority = priority


ACTS = [
    # UDIP
    ActRoute(
        act_name="Udip",
        aliases=[
            "udip",
            "dostęp do informacji", "dostep do informacji",
            "informacja publiczna", "informacji publicznej",
            "biuletyn informacji publicznej", "bip",
            "wniosek o informację", "wniosek o informacje",
            "informacja przetworzona", "przetworzona",
            "odmowa udostępnienia", "nieudostępnienie"
        ],
        priority=100
    ),

    # KPA
    ActRoute(
        act_name="Kodeks postępowania administracyjnego",
        aliases=[
            "kpa", "kodeks postępowania administracyjnego",
            "postępowanie administracyjne",
            "decyzja administracyjna",
            "odwołanie", "odwołanie od decyzji",
            "zażalenie",
            "wznowienie postępowania",
            "stwierdzenie nieważności", "nieważność decyzji",
            "ponaglenie",
            "milczące załatwienie", "milczace zalatwienie",
            "termin załatwienia sprawy", "termin zalatwienia sprawy",
            "metryka sprawy",
        ],
        priority=90
    ),

    # RODO UE / GDPR
    ActRoute(
        act_name="Rodo ue",
        aliases=[
            "gdpr",
            "ogólne rozporządzenie", "ogolne rozporzadzenie",
            "2016/679",
            "rozporządzenie 2016/679", "rozporzadzenie 2016/679",
            "profilowanie", "profilowania",
            "naruszenie ochrony danych", "naruszenia ochrony danych",
            "72h", "72 godz", "72 godziny", "72 hours",
            "administracyjna kara pieniężna", "kara pieniężna",
            "4% obrotu", "20 mln", "20 milionów"
        ],
        priority=110
    ),

    # RODO PL (ustawa)
    ActRoute(
        act_name="Rodo",
        aliases=[
            "uodo",
            "ustawa o ochronie danych",
            "prezes urzędu", "prezes urzedu",
            "puodo",
            "inspektor ochrony danych", "iod",
        ],
        priority=95
    ),

    # PPSA
    ActRoute(
        act_name="PPSA",
        aliases=[
            "ppsa",
            "sąd administracyjny", "sad administracyjny",
            "wsa", "wojewódzki sąd administracyjny", "wojewodzki sad administracyjny",
            "nsa", "naczelny sąd administracyjny", "naczelny sad administracyjny",
            "skarga do wsa",
            "skarga do sądu administracyjnego", "skarga do sadu administracyjnego",
            "skarga kasacyjna",
            "skarga na bezczynność", "skarga na bezczynnosc",
            "skarga na przewlekłość", "skarga na przewleklosc",
            "grzywna za bezczynność", "grzywna za bezczynnosc",
            "wstrzymanie wykonania decyzji",
            "odrzucenie skargi", "oddalenie skargi"
        ],
        priority=90
    ),

    # Prawo budowlane
    ActRoute(
        act_name="Prawo budowlane",
        aliases=[
            "prawo budowlane",
            "pozwolenie na budowę", "pozwolenie na budowe",
            "zgłoszenie budowy", "zgloszenie budowy",
            "roboty budowlane",
            "nadzór budowlany", "nadzor budowlany",
            "pinb", "winb",
            "samowola", "samowola budowlana",
            "legalizacja", "opłata legalizacyjna", "oplata legalizacyjna",
            "rozbiórka", "rozbiorka",
            "pozwolenie na użytkowanie", "pozwolenie na uzytkowanie",
            "zakończenie budowy", "zakonczenie budowy",
            "zmiana sposobu użytkowania", "zmiana sposobu uzytkowania",
            "kierownik budowy", "dziennik budowy",
            "katastrofa budowlana",
        ],
        priority=90
    ),
]


def route_act_names(query: str, max_acts: int = 2) -> List[str]:
    """
    Zwraca listę nazw aktów (act_name) najlepiej pasujących do zapytania.
    Działa deterministycznie, minimalizuje false-positive na krótkich aliasach.
    """
    q = (query or "").lower().strip()
    scores = []  # (score, act_name)

    for act in ACTS:
        score = 0
        for alias in act.aliases:
            if not alias:
                continue

            # krótkie aliasy (akronimy) -> dopasowanie na granicach słów
            if len(alias) <= 4:
                if re.search(rf"\b{re.escape(alias)}\b", q):
                    score += 15
            else:
                if alias in q:
                    # dłuższe frazy są bardziej precyzyjne
                    score += 8 + (len(alias) // 3)

        if score > 0:
            score += act.priority / 10
            scores.append((score, act.act_name))

    scores.sort(key=lambda x: x[0], reverse=True)

    # unikalne akty w kolejności najlepszych dopasowań
    out: List[str] = []
    seen = set()
    for _, name in scores:
        if name not in seen:
            out.append(name)
            seen.add(name)

    # lekkie dopalenie RODO/GDPR: jeśli padło "rodo" w pytaniu, często warto mieć GDPR w top
    if "rodo" in q or "gdpr" in q or "2016/679" in q:
        if "Rodo ue" not in out:
            out.insert(0, "Rodo ue")

    return out[:max_acts]

# src/routing.py
import re
from typing import List, Optional

# Prosta klasa do przechowywania reguł
class ActRoute:
    def __init__(self, act_name: str, aliases: List[str], priority: int = 0):
        self.act_name = act_name  # To musi pasować do pola 'act_name' w bazie Chroma
        self.aliases = aliases    # Słowa kluczowe
        self.priority = priority

# ---------------------------------------------------------
# KONFIGURACJA ROUTINGU (TWOJE USTAWY)
# ---------------------------------------------------------
# Ważne: act_name musi być identyczne jak to, co generuje vectorstore.py
# (czyli np. "Udip", "Kpa", "Rodo" - wielkość liter ma znaczenie!)

ACTS = [
    # 1. Ustawa o dostępie do informacji publicznej
    ActRoute(
        act_name="Udip", 
        aliases=[
            "udip", "dostęp do informacji", "dostep do informacji", 
            "informacja publiczna", "informacji publicznej",
            "biuletyn informacji publicznej", "bip",
            "wniosek o informację", "przetworzona",
            "nieudostępnienie", "odmowa udostępnienia"
        ],
        priority=100
    ),

    # 2. Kodeks postępowania administracyjnego
    ActRoute(
        act_name="Kodeks postępowania administracyjnego", # Lub "Kpa" - zależy jak vectorstore zapisał
        aliases=[
            "kpa", "kodeks postępowania administracyjnego", 
            "postępowanie administracyjne", "decyzja administracyjna", 
            "odwołanie", "zażalenie", "wznowienie postępowania",
            "organ administracji", "termin załatwienia sprawy",
            "milczące załatwienie", "bezczynność organu"
        ],
        priority=90
    ),
    
    # Obsługa skrótu Kpa (gdyby vectorstore zapisał skrótowo)
    ActRoute(
        act_name="Kpa", 
        aliases=["kpa", "kodeks postępowania administracyjnego"],
        priority=89
    ),

    # 3. GDPR (Rozporządzenie UE 2016/679)
ActRoute(
    act_name="Rodo ue",   # MUSI być identyczne jak w vectorstore / metadanych
    aliases=[
        "gdpr",
        "rozporządzenie 2016/679", "2016/679",
        "rozporzadzenie 2016/679",
        "rozporządzenie ue", "rozporzadzenie ue",
        "rozporządzenie o ochronie danych", "rozporzadzenie o ochronie danych",
        # typowe słowa-klucze, które od razu powinny preferować GDPR:
        "profilowanie", "profilowania",
        "naruszenie ochrony danych", "naruszenia ochrony danych",
        "zgłosić naruszenie", "zglosic naruszenie",
        "72 godz", "72h",
        "administracyjna kara pieniężna", "kara pieniężna", "kara pieniezna",
        "4% obrotu", "20 mln"
    ],
    priority=110  # wyżej niż krajowe "Rodo"
),

    # 3. RODO (Ustawa o ochronie danych osobowych)
    ActRoute(
        act_name="Rodo",
        aliases=[
            "rodo", "ochrona danych", "dane osobowe", "odo",
            "inspektor ochrony danych", "prezes urzędu", "uodo",
            "naruszenie ochrony danych", "przetwarzanie danych"
        ],
        priority=95
    )
]

def _norm(text: str) -> str:
    """Prosta normalizacja tekstu."""
    return text.lower().strip()

def route_act_names(query: str, max_acts: int = 2) -> List[str]:
    """
    Analizuje zapytanie i zwraca listę nazw aktów prawnych,
    które najbardziej pasują do tematu.
    """
    q = _norm(query)
    scores = []

    for act in ACTS:
        score = 0
        for alias in act.aliases:
            # Sprawdzamy czy alias występuje w zapytaniu
            if alias in q:
                # Dłuższe aliasy są zazwyczaj bardziej precyzyjne, więc punktujemy je wyżej
                score += 10 + len(alias)
        
        if score > 0:
            scores.append((score + act.priority, act.act_name))

    # Sortujemy od najlepszego dopasowania
    scores.sort(key=lambda x: x[0], reverse=True)

    # Zwracamy top N wyników
    unique_acts = []
    seen = set()
    for _, name in scores:
        if name not in seen:
            unique_acts.append(name)
            seen.add(name)
    
    if ("rodo" in q) or ("gdpr" in q) or ("2016/679" in q):
        # dołóż GDPR
        if "Rodo ue" not in unique_acts:
            unique_acts.insert(0, "Rodo ue")
        # dołóż ustawę krajową
        if "Rodo" not in unique_acts:
            unique_acts.append("Rodo")
    
    return unique_acts[:max_acts]
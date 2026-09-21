"""Generazione automatica delle domande da porre al personaggio.

Serve a coprire le sei categorie previste in configurazione, che la versione
precedente ignorava: generava solo `philosophy_values` e `anachronistic`,
entrambe cablate nel codice.

Due modalita', combinabili:

- **estrattiva**, senza rete ne' LLM: le domande nascono dal corpus stesso
  (entita' nominate, valori misurati, campi semantici) tramite schemi per
  categoria. E' quella che garantisce l'automazione anche offline;
- **generativa**, se un LLM e' raggiungibile: domande piu' naturali e varie.

Se l'LLM non risponde si ricade sull'estrattiva, quindi la pipeline non si
ferma mai per l'indisponibilita' di un servizio.
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..data_models import CompleteStyleProfile, TextSegment

logger = logging.getLogger(__name__)

# Le sei categorie previste dalla configurazione.
CATEGORY_HISTORICAL = "historical_facts"
CATEGORY_VALUES = "philosophy_values"
CATEGORY_ANACHRONISTIC = "anachronistic"
CATEGORY_PERSONALITY = "character_personality"
CATEGORY_EXPERTISE = "domain_expertise"
CATEGORY_CASUAL = "casual_conversation"

ALL_CATEGORIES = (
    CATEGORY_HISTORICAL, CATEGORY_VALUES, CATEGORY_ANACHRONISTIC,
    CATEGORY_PERSONALITY, CATEGORY_EXPERTISE, CATEGORY_CASUAL,
)

# Schemi in italiano: la lingua in cui l'utente interroga il personaggio.
# La risposta restera' nello stile misurato sul corpus, che puo' essere in
# un'altra lingua: e' il comportamento voluto.
_TEMPLATES: Dict[str, List[str]] = {
    CATEGORY_HISTORICAL: [
        "Che cosa accadde a {entity}?",
        "Puoi raccontarmi di {entity}?",
        "Qual e' stato il tuo ruolo in merito a {entity}?",
        "Come giudichi la vicenda di {entity}?",
        "Perche' {entity} ebbe l'importanza che ebbe?",
    ],
    CATEGORY_VALUES: [
        "Che cosa significa per te {value}?",
        "Perche' tieni tanto a {value}?",
        "Puo' esserci {value} senza potere?",
        "Quando hai dovuto scegliere fra {value} e la convenienza, che cosa hai fatto?",
        "Un uomo privo di {value} merita rispetto?",
    ],
    CATEGORY_ANACHRONISTIC: [
        "Che cosa pensi di {modern_thing}?",
        "Come useresti {modern_thing} al tuo posto?",
        "Conosci {modern_thing}?",
        "Ti spiego cos'e' {modern_thing}: che ne diresti?",
    ],
    CATEGORY_PERSONALITY: [
        "Di che cosa hai paura?",
        "Qual e' stato l'errore che rimpiangi di piu'?",
        "Chi ti ha insegnato di piu', e che cosa?",
        "Come vorresti essere ricordato?",
        "C'e' qualcosa che non hai mai detto a nessuno?",
        "Che cosa ti fa perdere la pazienza?",
        "Ti sei mai sentito solo nella tua posizione?",
    ],
    CATEGORY_EXPERTISE: [
        "Come si affronta una questione di {field}?",
        "Che consiglio daresti a chi si occupa di {field}?",
        "Qual e' l'errore piu' comune in materia di {field}?",
        "Che cosa distingue chi eccelle in {field}?",
    ],
    CATEGORY_CASUAL: [
        "Come hai passato la giornata?",
        "Che cosa mangi di solito?",
        "Dormi bene?",
        "Che tempo fa dalle tue parti?",
        "Hai amici veri?",
        "Come ti riposi quando puoi?",
    ],
}

# Concetti moderni per le domande anacronistiche: il modello deve imparare a
# non riconoscerli, non a spiegarli.
MODERN_CONCEPTS = [
    "internet", "il telefono cellulare", "l'intelligenza artificiale",
    "l'automobile", "l'aereo", "la televisione", "il computer",
    "i social network", "il vaccino", "l'elettricita'", "la fotografia",
    "il treno", "la bomba atomica", "la democrazia parlamentare moderna",
    "il capitalismo finanziario", "i diritti umani universali",
    "la teoria dell'evoluzione", "il motore a scoppio", "la radio",
]

# Parole che sembrano entita' ma non lo sono: inizio frase, titoli generici.
_ENTITY_STOPLIST = {
    "il", "lo", "la", "i", "gli", "le", "un", "una", "questo", "quello",
    "the", "and", "but", "his", "her", "their", "this", "that",
}

_ENTITY = re.compile(r"\b([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]{3,})\b")


@dataclass
class Question:
    text: str
    category: str
    source_segment_id: Optional[str] = None
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuestionGenerator:
    """Produce le domande che alimentano la generazione del dataset."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_provider=None,
        seed: Optional[int] = None,
    ):
        self.config = config or {}
        self.llm = llm_provider
        self.random = random.Random(
            seed if seed is not None else self.config.get("advanced", {}).get("seed", 42)
        )

    # ------------------------------------------------------------- pubblico

    def generate(
        self,
        profile: CompleteStyleProfile,
        segments: Sequence[TextSegment],
        total: int,
        distribution: Optional[Dict[str, float]] = None,
    ) -> List[Question]:
        """Genera `total` domande distribuite fra le categorie configurate."""
        distribution = distribution or self.config.get("dataset", {}).get(
            "conversation_types", {}
        )
        if not distribution:
            distribution = {category: 1 / len(ALL_CATEGORIES) for category in ALL_CATEGORIES}

        counts = self._allocate(total, distribution)
        questions: List[Question] = []

        for category, count in counts.items():
            if count <= 0:
                continue
            generated = self._generate_category(category, count, profile, segments)
            if len(generated) < count:
                logger.info(
                    f"Categoria '{category}': {len(generated)}/{count} domande "
                    "(materiale insufficiente nel corpus)"
                )
            questions.extend(generated)

        self.random.shuffle(questions)
        return questions

    # ------------------------------------------------------------ ripartizione

    @staticmethod
    def _allocate(total: int, distribution: Dict[str, float]) -> Dict[str, int]:
        """Ripartisce `total` fra le categorie rispettando le proporzioni.

        La divisione lascia sempre un resto: assegnarlo tutto alla categoria
        col peso maggiore la gonfia oltre la sua quota. Si usa il metodo dei
        resti maggiori — una unita' a ciascuna delle categorie con la frazione
        piu' alta — cosi' la somma torna esattamente a `total` e nessuna
        categoria si allontana di piu' di un esempio dalla proporzione voluta.
        """
        weight_sum = sum(distribution.values()) or 1.0

        exact = {
            category: total * weight / weight_sum
            for category, weight in distribution.items()
        }
        counts = {category: int(value) for category, value in exact.items()}

        # I resti si assegnano alle categorie con la frazione piu' alta, una
        # ciascuna: darli tutti alla categoria col peso maggiore la gonfiava
        # oltre la sua quota (su 18 esempi, 8 invece di 4 a `historical_facts`).
        remainder = total - sum(counts.values())
        if remainder > 0:
            by_fraction = sorted(
                exact, key=lambda c: (exact[c] - int(exact[c]), distribution[c]), reverse=True
            )
            for category in by_fraction[:remainder]:
                counts[category] += 1

        return counts

    # ------------------------------------------------------------ categorie

    def _generate_category(
        self,
        category: str,
        count: int,
        profile: CompleteStyleProfile,
        segments: Sequence[TextSegment],
    ) -> List[Question]:
        if category == CATEGORY_HISTORICAL:
            return self._historical(count, segments, self._author_tokens(profile))
        if category == CATEGORY_VALUES:
            return self._values(count, profile)
        if category == CATEGORY_ANACHRONISTIC:
            return self._anachronistic(count, profile)
        if category == CATEGORY_EXPERTISE:
            return self._expertise(count, profile)
        if category == CATEGORY_PERSONALITY:
            return self._from_templates(count, CATEGORY_PERSONALITY, {})
        if category == CATEGORY_CASUAL:
            return self._from_templates(count, CATEGORY_CASUAL, {})

        logger.warning(f"Categoria non riconosciuta, ignorata: {category}")
        return []

    def _author_tokens(self, profile: CompleteStyleProfile) -> set[str]:
        """Forme del nome del personaggio, da non usare come entita'.

        Chiedere a Cesare «che cosa accadde a Caesar?» produce una domanda
        priva di senso, e su un corpus dove l'autore parla di se' in terza
        persona il suo nome e' proprio l'entita' piu' frequente.
        """
        tokens = {
            part.lower()
            for part in re.split(r"\s+", profile.author_name or "")
            if len(part) > 2
        }
        # Le forme attestate nel corpus possono differire dal nome fornito
        # (Cesare/Caesar): le auto-citazioni raccolte in analisi le contengono.
        for reference in profile.rhetoric.self_reference_patterns[:5]:
            for word in re.findall(r"[A-ZÀ-ÖØ-Þ][\w'-]{3,}", reference):
                tokens.add(word.lower())
        return tokens

    def _historical(
        self,
        count: int,
        segments: Sequence[TextSegment],
        exclude: Optional[set] = None,
    ) -> List[Question]:
        """Domande ancorate a passaggi reali del corpus.

        Ogni domanda porta con se' il passo da cui nasce: e' cio' che
        permette alla generazione successiva di rispondere con fatti
        attestati invece che inventati.
        """
        usable = [s for s in segments if s.char_count > 300]
        if not usable:
            return []

        questions: List[Question] = []
        templates = _TEMPLATES[CATEGORY_HISTORICAL]

        # Campionamento senza rimpiazzo finche' il corpus lo consente: evita
        # di interrogare venti volte lo stesso passaggio.
        pool = list(usable)
        self.random.shuffle(pool)

        index = 0
        while len(questions) < count and pool:
            segment = pool[index % len(pool)]
            index += 1
            if index > count * 3:
                break

            entities = self._extract_entities(segment.content, exclude)
            if not entities:
                continue

            entity = self.random.choice(entities)
            template = templates[len(questions) % len(templates)]
            questions.append(Question(
                text=template.format(entity=entity),
                category=CATEGORY_HISTORICAL,
                source_segment_id=segment.id,
                context=segment.content[:3000],
                metadata={"entity": entity, "source_file": segment.source_file},
            ))

        return questions

    def _values(self, count: int, profile: CompleteStyleProfile) -> List[Question]:
        values = profile.values.dominant_values
        if not values:
            return []

        questions: List[Question] = []
        templates = _TEMPLATES[CATEGORY_VALUES]

        for i in range(count):
            value = values[i % len(values)]
            template = templates[i % len(templates)]
            examples = profile.values.example_passages.get(value, [])
            questions.append(Question(
                text=template.format(value=value),
                category=CATEGORY_VALUES,
                context=examples[0] if examples else "",
                metadata={"value": value},
            ))
        return questions

    def _anachronistic(self, count: int, profile: CompleteStyleProfile) -> List[Question]:
        concepts = list(MODERN_CONCEPTS)
        if profile.knowledge and profile.knowledge.anachronistic_concepts:
            # I concetti dichiarati in configurazione vengono prima: sono
            # quelli che l'utente ha indicato come rilevanti per il personaggio.
            concepts = list(profile.knowledge.anachronistic_concepts) + concepts

        questions: List[Question] = []
        templates = _TEMPLATES[CATEGORY_ANACHRONISTIC]

        for i in range(count):
            concept = concepts[i % len(concepts)]
            template = templates[i % len(templates)]
            questions.append(Question(
                text=template.format(modern_thing=concept),
                category=CATEGORY_ANACHRONISTIC,
                metadata={"concept": concept},
            ))
        return questions

    def _expertise(self, count: int, profile: CompleteStyleProfile) -> List[Question]:
        fields = list(profile.vocabulary.semantic_field_distribution.keys())
        if not fields:
            # Senza campi semantici si ripiega sui termini distintivi, che
            # indicano comunque di che cosa l'autore si occupa.
            fields = [t["term"] for t in profile.vocabulary.distinctive_terms[:8]]
        if not fields:
            return []

        questions: List[Question] = []
        templates = _TEMPLATES[CATEGORY_EXPERTISE]

        for i in range(count):
            field_name = fields[i % len(fields)]
            template = templates[i % len(templates)]
            questions.append(Question(
                text=template.format(field=field_name),
                category=CATEGORY_EXPERTISE,
                metadata={"field": field_name},
            ))
        return questions

    def _from_templates(
        self, count: int, category: str, values: Dict[str, str]
    ) -> List[Question]:
        templates = _TEMPLATES[category]
        questions: List[Question] = []
        for i in range(count):
            template = templates[i % len(templates)]
            questions.append(Question(
                text=template.format(**values) if values else template,
                category=category,
            ))
        return questions

    # -------------------------------------------------------------- entita'

    def _extract_entities(self, text: str, exclude: Optional[set] = None) -> List[str]:
        """Nomi propri plausibili: maiuscola non a inizio frase.

        Euristica volutamente semplice, perche' deve funzionare anche senza
        modello NER (latino e greco antico non ne hanno uno affidabile).
        """
        candidates: List[str] = []
        for sentence in re.split(r"[.!?;]\s+", text):
            words = sentence.split()
            # Si salta la prima parola: la maiuscola li' e' solo ortografia.
            for word in words[1:]:
                match = _ENTITY.match(word.strip(".,;:!?\"'()"))
                if not match:
                    continue
                entity = match.group(1)
                lowered = entity.lower()
                if lowered in _ENTITY_STOPLIST:
                    continue
                if exclude and _resembles_any(lowered, exclude):
                    continue
                candidates.append(entity)

        # Le entita' ripetute sono le piu' rilevanti del passaggio.
        counter: Dict[str, int] = {}
        for candidate in candidates:
            counter[candidate] = counter.get(candidate, 0) + 1

        return [
            entity for entity, _ in
            sorted(counter.items(), key=lambda kv: -kv[1])[:10]
        ]

    # ----------------------------------------------------------------- LLM

    def refine_with_llm(
        self, questions: List[Question], profile: CompleteStyleProfile
    ) -> List[Question]:
        """Riformula le domande in modo piu' naturale, se un LLM e' presente.

        Un fallimento qui non e' un errore: le domande estrattive restano
        valide e la pipeline prosegue con quelle.
        """
        if not self.llm or not questions:
            return questions

        batch = [q.text for q in questions[:50]]
        prompt = (
            f"Queste domande vanno poste a {profile.author_name}"
            f"{f' ({profile.era})' if profile.era else ''}.\n"
            "Riscrivile in modo che suonino naturali, come le porrebbe una persona "
            "curiosa in una conversazione. Mantieni lo stesso argomento e lo stesso "
            "numero di domande, nello stesso ordine.\n\n"
            + "\n".join(f"{i + 1}. {text}" for i, text in enumerate(batch))
            + '\n\nRispondi con un oggetto JSON: {"questions": ["...", "..."]}'
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="Riformuli domande mantenendone il significato. Rispondi solo JSON.",
                json_mode=True,
            )
            data = json.loads(_strip_code_fence(response))
            refined = data.get("questions", []) if isinstance(data, dict) else []
        except Exception as exc:
            logger.warning(f"Riformulazione con LLM non riuscita: {exc}")
            return questions

        for question, new_text in zip(questions, refined):
            if isinstance(new_text, str) and len(new_text.strip()) > 8:
                question.text = new_text.strip()

        return questions


def _strip_code_fence(text: str) -> str:
    """Rimuove i delimitatori di blocco che molti modelli aggiungono comunque."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _resembles_any(word: str, names: set) -> bool:
    """Vero se la parola e' una variante di uno dei nomi indicati.

    Il nome fornito dall'utente e quello attestato nel corpus raramente
    coincidono lettera per lettera — si scrive «Cesare» e il testo latino dice
    *Caesar* — quindi il confronto esatto lascerebbe passare proprio il caso
    che si vuole escludere.
    """
    if word in names:
        return True

    try:
        from rapidfuzz import fuzz
    except ImportError:
        # Senza rapidfuzz resta il confronto per prefisso, che copre le
        # desinenze ma non i cambi di grafia interni.
        return any(word.startswith(n[:4]) or n.startswith(word[:4]) for n in names if len(n) >= 4)

    return any(fuzz.ratio(word, name) >= 80 for name in names)

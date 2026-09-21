"""Sintesi del prompt di sistema a partire dal profilo misurato.

Il generatore precedente produceva tre righe:

    You are {author}.
    Speak using sentences of approx {n} words.
    Emphasize these values: {...}

cioe' buttava via quasi tutto cio' che la pipeline aveva estratto — e i
valori erano comunque vuoti per via del bug sui percorsi. Qui ogni dimensione
misurata diventa un'istruzione operativa, perche' un modello segue "apri
spesso con *Quibus rebus*" molto meglio di "usa frasi di 23,4 parole".

Il prompt e' costruito in modo deterministico dai numeri: nessuna chiamata a
un LLM, quindi funziona anche completamente offline. Se un LLM e' configurato,
`refine_with_llm` puo' arricchire la descrizione, ma non e' necessario.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..data_models import CompleteStyleProfile, SupportedLanguage

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    SupportedLanguage.ITALIAN: "italiano",
    SupportedLanguage.ENGLISH: "inglese",
    SupportedLanguage.FRENCH: "francese",
    SupportedLanguage.GERMAN: "tedesco",
    SupportedLanguage.SPANISH: "spagnolo",
    SupportedLanguage.PORTUGUESE: "portoghese",
    SupportedLanguage.RUSSIAN: "russo",
    SupportedLanguage.UKRAINIAN: "ucraino",
    SupportedLanguage.HUNGARIAN: "ungherese",
    SupportedLanguage.GREEK_MODERN: "greco moderno",
    SupportedLanguage.GREEK_ANCIENT: "greco antico",
    SupportedLanguage.LATIN: "latino",
    SupportedLanguage.UNKNOWN: "lingua non determinata",
}

PERSPECTIVE_INSTRUCTIONS = {
    "first_person": "Parla in prima persona ('io'), come nelle tue opere.",
    "third_person_self_reference": (
        "Riferisciti a te stesso in TERZA PERSONA usando il tuo nome, come fai "
        "nei tuoi scritti. Non dire 'io ho fatto': di' '{author} fece'. "
        "E' il tratto piu' riconoscibile della tua voce: mantienilo sempre."
    ),
    "mixed": "Alterna prima e terza persona come nelle tue opere.",
    "third_person": "Mantieni un'esposizione impersonale, in terza persona.",
}

REGISTER_INSTRUCTIONS = {
    "formal_expository": "Registro alto ed espositivo: esponi, non conversi.",
    "formal_dialogic": "Registro alto ma dialogico: interpella l'interlocutore.",
    "conversational_dialogic": "Registro colloquiale: procedi per domande e risposte.",
    "plain_narrative": "Registro piano e narrativo: racconta con parole comuni.",
}

ARGUMENT_INSTRUCTIONS = {
    "causal": "costruisci le risposte per nesso causale (causa → conseguenza)",
    "contrast": "procedi per contrapposizione, mettendo a confronto due termini",
    "concession": "concedi prima l'obiezione, poi ribatti",
    "authority": "appoggia le affermazioni all'autorita' e alla tradizione",
    "example": "chiarisci con esempi concreti",
    "condition": "ragiona per ipotesi e condizioni",
    "enumeration": "ordina il discorso per punti successivi",
}


class PersonaSynthesizer:
    """Traduce il profilo misurato in istruzioni per il modello."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    # ------------------------------------------------------------- pubblico

    def build_system_prompt(self, profile: CompleteStyleProfile) -> str:
        override = self.config.get("persona", {}).get("system_prompt_override")
        if override:
            return str(override)

        author = profile.author_name or "il personaggio"
        sections: List[str] = [self._identity(profile, author)]

        for block in (
            self._voice_section(profile, author),
            self._syntax_section(profile),
            self._vocabulary_section(profile),
            self._reasoning_section(profile),
            self._values_section(profile),
            self._boundaries_section(profile),
        ):
            if block:
                sections.append(block)

        return "\n\n".join(sections).strip()

    def build_training_guidelines(self, profile: CompleteStyleProfile) -> Dict[str, List[str]]:
        """Regole operative, usate per guidare e valutare la generazione."""
        guidelines: Dict[str, List[str]] = {"do": [], "avoid": []}

        syntax = profile.syntax
        if syntax.avg_sentence_length:
            low = max(4, int(syntax.avg_sentence_length - syntax.sentence_length_std))
            high = int(syntax.avg_sentence_length + syntax.sentence_length_std)
            guidelines["do"].append(
                f"Mantieni le frasi fra {low} e {high} parole (media {syntax.avg_sentence_length})."
            )

        if syntax.morphology_available:
            if syntax.subordination_ratio >= 0.5:
                guidelines["do"].append(
                    "Usa periodi con subordinate: oltre meta' delle frasi del corpus le contiene."
                )
            elif syntax.subordination_ratio <= 0.2:
                guidelines["avoid"].append(
                    "Evita i periodi lunghi e subordinati: il corpus procede per frasi semplici."
                )

        # Le formule vanno indicate come *disponibili*, non come obbligatorie.
        # Un'istruzione imperativa ("apri le frasi con…") viene applicata
        # letteralmente: su un dataset generato si e' visto il modello aprire
        # 15 risposte su 18 con la stessa formula. Addestrare su quel materiale
        # insegna un tic, non uno stile.
        openers = [o["opener"] for o in profile.voice.signature_openers[:8]]
        if openers:
            guidelines["do"].append(
                "Attingi alle tue aperture tipiche ("
                + ", ".join(f"'{o}'" for o in openers)
                + ") con parsimonia: al massimo in una risposta su quattro, "
                "e mai due volte di seguito."
            )

        formulas = profile.vocabulary.formulas_and_collocations[:8]
        if formulas:
            guidelines["do"].append(
                "Le formule attestate ("
                + ", ".join(f"'{f}'" for f in formulas)
                + ") sono a disposizione quando il discorso le richiede, "
                "non da inserire in ogni risposta."
            )

        guidelines["avoid"].append(
            "Non aprire ogni risposta allo stesso modo: la ripetizione "
            "meccanica di una formula non e' stile, e' un tic."
        )

        if profile.rhetoric.narrative_perspective == "third_person_self_reference":
            guidelines["avoid"].append(
                "Non usare mai la prima persona per parlare di te: il corpus usa la terza."
            )

        if profile.knowledge and profile.knowledge.anachronistic_concepts:
            guidelines["avoid"].append(
                "Non riconoscere ne' spiegare: "
                + ", ".join(profile.knowledge.anachronistic_concepts[:10]) + "."
            )

        guidelines["avoid"].append(
            "Non usare lessico o concetti posteriori alla tua epoca, nemmeno per analogia."
        )
        return guidelines

    # ------------------------------------------------------------- sezioni

    @staticmethod
    def _identity(profile: CompleteStyleProfile, author: str) -> str:
        language = LANGUAGE_NAMES.get(profile.primary_language, "lingua non determinata")
        lines = [f"Sei {author}."]
        if profile.era:
            lines.append(f"Vivi nell'epoca: {profile.era}.")
        lines.append(
            f"Il tuo corpus di riferimento e' in {language}: "
            "rispondi nella lingua in cui ti si rivolge, conservando pero' il tuo modo di esprimerti."
        )
        stats = profile.corpus_stats
        if stats.total_words:
            lines.append(
                f"Il tuo stile e' ricavato da {stats.total_words:,} parole "
                f"tratte da {stats.total_files} font{'i' if stats.total_files != 1 else 'e'}."
                .replace(",", ".")
            )
        return "\n".join(lines)

    @staticmethod
    def _voice_section(profile: CompleteStyleProfile, author: str) -> str:
        voice = profile.voice
        lines = ["## Come parli"]

        perspective = PERSPECTIVE_INSTRUCTIONS.get(profile.rhetoric.narrative_perspective)
        if perspective:
            lines.append("- " + perspective.format(author=author))

        register = REGISTER_INSTRUCTIONS.get(voice.speech_register)
        if register:
            lines.append(f"- {register}")

        openers = [o["opener"] for o in voice.signature_openers[:10]]
        if openers:
            # Elencare gli incipit come tratti propri equivale a ordinarne
            # l'uso: misurato su un dataset generato, 17 risposte su 18
            # aprivano con la stessa formula. Un corpus ristretto rende il
            # problema estremo, perche' il profilo registra pochissime
            # aperture e il modello le prende per obbligatorie.
            lines.append(
                "- Fra le tue aperture ricorrono "
                + ", ".join(f"«{o}»" for o in openers)
                + ": usale quando cadono a proposito, non a ogni risposta. "
                "Variare l'attacco fa parte della tua voce quanto le formule."
            )

        vocatives = [v["term"] for v in voice.vocatives[:6]]
        if vocatives:
            lines.append(
                "- Ti rivolgi all'interlocutore chiamandolo: " + ", ".join(vocatives) + "."
            )

        if voice.question_ratio >= 0.10:
            lines.append(
                f"- Poni domande di frequente ({voice.question_ratio:.0%} delle frasi): "
                "usale per condurre il discorso."
            )
        elif voice.question_ratio <= 0.02:
            lines.append("- Non fai quasi mai domande: affermi.")

        if voice.exclamation_ratio >= 0.05:
            lines.append("- Ricorri all'esclamazione per dare enfasi.")

        if voice.modality_distribution:
            dominant = max(voice.modality_distribution, key=voice.modality_distribution.get)
            modality_text = {
                "certainty": "Ti esprimi con certezza: asserisci senza attenuare.",
                "doubt": "Attenui spesso le affermazioni: «forse», «pare che».",
                "obligation": "Parli in termini di dovere e necessita'.",
            }.get(dominant)
            if modality_text:
                lines.append(f"- {modality_text}")

        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _syntax_section(profile: CompleteStyleProfile) -> str:
        syntax = profile.syntax
        if not syntax.avg_sentence_length:
            return ""

        lines = ["## Il ritmo del tuo periodo"]
        lines.append(
            f"- Frase media: {syntax.avg_sentence_length} parole "
            f"(variabilita' {syntax.sentence_length_std})."
        )

        # Senza modello morfologico questi rapporti sono 0 per assenza di
        # misura: affermare "periodo paratattico" sarebbe inventare un tratto.
        if syntax.morphology_available:
            if syntax.subordination_ratio >= 0.5:
                lines.append(
                    f"- Periodo ipotattico: il {syntax.subordination_ratio:.0%} delle frasi "
                    "contiene subordinate. Costruisci per incassi successivi."
                )
            elif syntax.subordination_ratio <= 0.25:
                lines.append(
                    "- Periodo paratattico: frasi brevi e giustapposte, poche subordinate."
                )

            if syntax.coordination_ratio >= 0.5:
                lines.append("- Coordini molto: lega le proposizioni in catena.")

        person = syntax.person_distribution or {}
        if person:
            dominant = max(person, key=person.get)
            labels = {"1": "prima", "2": "seconda", "3": "terza"}
            lines.append(
                f"- I verbi sono prevalentemente di {labels.get(dominant, dominant)} persona "
                f"({person[dominant]:.0%})."
            )

        special = syntax.special_patterns or {}
        long_ratio = special.get("long_sentences_over_40_words", 0)
        total = special.get("total_sentences", 0)
        if total and long_ratio / total > 0.15:
            lines.append(
                "- Non temere i periodi molto lunghi: ne usi in abbondanza."
            )

        return "\n".join(lines)

    @staticmethod
    def _vocabulary_section(profile: CompleteStyleProfile) -> str:
        vocab = profile.vocabulary
        terms = [t["term"] for t in vocab.distinctive_terms[:25]]
        formulas = vocab.formulas_and_collocations[:12]

        if not terms and not formulas:
            return ""

        lines = ["## Il tuo lessico"]
        if terms:
            lines.append("- Parole che ricorrono in tutta la tua opera: " + ", ".join(terms) + ".")
        if formulas:
            lines.append(
                "- Formule attestate nei tuoi scritti: "
                + ", ".join(f"«{f}»" for f in formulas)
                + ". Sono un repertorio, non un ritornello: chi le ripete a ogni "
                "frase non ti somiglia, ti imita male."
            )

        if vocab.semantic_field_distribution:
            ordered = sorted(
                vocab.semantic_field_distribution.items(), key=lambda kv: -kv[1]
            )[:4]
            lines.append(
                "- I tuoi campi di discorso: "
                + ", ".join(f"{name} ({pct:.0f}%)" for name, pct in ordered) + "."
            )

        if vocab.type_token_ratio:
            if vocab.type_token_ratio >= 0.20:
                lines.append("- Il tuo lessico e' vario: eviti di ripetere le stesse parole.")
            elif vocab.type_token_ratio <= 0.08:
                lines.append(
                    "- Il tuo lessico e' volutamente ristretto: torni sulle stesse parole chiave."
                )

        return "\n".join(lines)

    @staticmethod
    def _reasoning_section(profile: CompleteStyleProfile) -> str:
        distribution = profile.rhetoric.argument_type_distribution
        if not distribution:
            return ""

        ordered = sorted(distribution.items(), key=lambda kv: -kv[1])[:3]
        instructions = [
            ARGUMENT_INSTRUCTIONS[name]
            for name, _ in ordered if name in ARGUMENT_INSTRUCTIONS
        ]
        if not instructions:
            return ""

        lines = ["## Come ragioni"]
        lines.append("- Quando argomenti, " + "; ".join(instructions) + ".")
        top_name, top_pct = ordered[0]
        lines.append(
            f"- La mossa argomentativa che ti e' piu' propria e' «{top_name}» "
            f"({top_pct:.0f}% dei connettivi che usi)."
        )
        return "\n".join(lines)

    @staticmethod
    def _values_section(profile: CompleteStyleProfile) -> str:
        values = profile.values
        if not values.dominant_values:
            return ""

        lines = ["## I tuoi valori"]
        for name in values.dominant_values[:6]:
            weight = values.value_distribution.get(name, 0)
            description = next(
                (c.get("description", "") for c in values.theme_clusters if c.get("value") == name),
                "",
            )
            entry = f"- **{name}** ({weight:.0f}% del peso valoriale)"
            if description:
                entry += f": {description}"
            lines.append(entry)

        lines.append(
            "Giudica uomini e situazioni a partire da questi valori: sono il criterio "
            "con cui distingui cio' che e' lodevole da cio' che non lo e'."
        )
        return "\n".join(lines)

    @staticmethod
    def _boundaries_section(profile: CompleteStyleProfile) -> str:
        knowledge = profile.knowledge
        lines = ["## I confini di cio' che sai"]

        if knowledge and knowledge.era_start and knowledge.era_start != "Unknown":
            lines.append(
                f"- La tua conoscenza si ferma a {knowledge.era_end or knowledge.era_start}."
            )
        if knowledge and knowledge.known_topics:
            lines.append("- Conosci bene: " + ", ".join(knowledge.known_topics[:12]) + ".")
        if knowledge and knowledge.anachronistic_concepts:
            lines.append(
                "- Non conosci e non puoi riconoscere: "
                + ", ".join(knowledge.anachronistic_concepts[:12]) + "."
            )

        lines.append(
            "- Se ti si parla di cose posteriori alla tua epoca, non fingere di capirle: "
            "chiedi spiegazioni o interpretale con le categorie del tuo tempo."
        )
        lines.append(
            "- Non citare mai fonti, eventi o persone successivi alla tua morte."
        )
        return "\n".join(lines)
